import gc
import itertools
import json
import logging as log
import html
import pickle
import re

import numpy as np
import zipfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import MutableMapping
from pathlib import Path

from tqdm import tqdm

from src.data.checksum import hash_files
from src.data.dataset_iterator import DatasetIterator
from src.data.encoding_utils import convert_time_format, markdown_to_text
from src.data.recency_cache import RecencyCacheDict

BOD = '<BOD>'
EOD = '<EOD>'
SIGNATURE = '<SIGNATURE>'
AGENT_FNAME = '<AGENT_FNAME>'
AGENT_NAME = '<AGENT_NAME>'
CUSTOMER_FNAME = '<CUSTOMER_FNAME>'
CUSTOMER_NAME = '<CUSTOMER_NAME>'


class Macro:
    def __init__(self, macro_id, title, active, restriction_type=None):
        self.id = macro_id
        self.title = title
        self.active = active
        self.restriction_type = restriction_type

    def __str__(self):
        return f"Macro({self.id}): {self.title}"

    def __repr__(self):
        return str(self)


class Author:
    def __init__(self, author_id, role, name=None, alias=None, email=None, signature=None, timezone=None):
        self.id = author_id
        self.name = name
        self.alias = alias
        self.email = email
        self.signature = signature
        self.role = role
        self.timezone = timezone

    def __str__(self):
        return f"[{self.role}]({self.name})"

    def __repr__(self):
        return str(self)


class Comment:
    def __init__(self, text, created_timestamp, author, macros=[], language='en', comment_type='public', group_id=None):
        self.author = author
        self.timestamp = created_timestamp
        self.group_id = group_id  # This is the group_id so far for this comment
        self.text = text
        self.language = language
        self.type = comment_type
        self.macros = macros

    def __str__(self):
        text = f"\n{self.author} at {self.timestamp}({self.language}, {self.type}): {self.text}\n"
        for macro in self.macros:
            text += str(macro)
        return text

    def __repr__(self):
        return str(self)


class Ticket:
    def __init__(self, ticket_id, group_id, subject, timestamp, comments, subdomain, channel='email'):
        self.id = ticket_id
        self.group_id = group_id
        self.subject = subject
        self.timestamp = timestamp
        self.comments = comments
        self.channel = channel
        self.subdomain = subdomain
        self.fill_missing_values()

    def __str__(self):
        text = '-' * 20 + '\n'
        text += f"At {self.timestamp} via {self.channel} with Subject: {self.subject}"
        for comment in self.comments:
            text += str(comment)
        text += '-' * 20 + '\n'
        return text

    def fill_missing_values(self):
        timezones = Counter([comment.author.timezone for comment in self.comments])

        for comment in self.comments:
            if comment.author.timezone is None:
                comment.author.timezone = timezones.most_common(1)[0][0]


class ZipFolderJsonFilesDict(MutableMapping):
    def __init__(self, zipfile_path, folder_path_suffix, key_typecast_fn=int, extension='.json', cache_size=1000,
                 load_in_memory=False):
        self.zipfile_path = zipfile_path
        self.load_in_memory = load_in_memory
        self.cache = {}

        with zipfile.ZipFile(self.zipfile_path, mode='r', allowZip64=True) as zip_file:
            self.files = OrderedDict()
            for file_path in sorted(zip_file.namelist()):
                if file_path.endswith(extension) and folder_path_suffix in file_path:
                    key = key_typecast_fn(file_path.split('/')[-1][:-(len(extension))])
                    self.files[key] = file_path
                    if self.load_in_memory:
                        try:
                            self.cache[key] = json.loads(zip_file.read(file_path))
                        except Exception:
                            continue  # if there is issue with loading a file, don't load it at all

        self.zip_file = zipfile.ZipFile(self.zipfile_path, mode='r', allowZip64=True)
        if not self.load_in_memory:
            self.cache = RecencyCacheDict(max_size=cache_size)

    def __contains__(self, item):
        return item in self.files

    def __getitem__(self, key):
        if key not in self.cache:
            path = self.files[key]
            value = None
            num_tries = 0
            # ensure that it is not stuck in infinite loop
            while value is None and num_tries < 5:
                try:
                    value = json.loads(self.zip_file.read(path))
                except:
                    num_tries += 1
                    self.zip_file.close()
                    self.zip_file = zipfile.ZipFile(self.zipfile_path, mode='r', allowZip64=True)
                    continue
            self.cache[key] = value

        return self.cache[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, v):
        raise NotImplementedError

    def __iter__(self):
        for key in self.files:
            yield key

    def __len__(self):
        return len(self.files)

    def __del__(self):
        try:
            self.zip_file.close()
        except Exception:
            pass


class ZippedZendeskDataset:
    ZENDESK_TIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

    def __init__(self, path, cache_size=1000, load=True,  extract_macros=False, cache_dir='/tmp'):
        self.path = Path(path)
        self.cache_size = cache_size
        self.macro_extract = extract_macros
        self.tickets, self.audits, self.users, self.macros, self.brands, self.groups = None, None, None, None, None, None
        self.idx2ticketid = None
        self.augmented_agent_ids = []
        # TODO(abhi) Remove language detection from the ticket, although this is the reason for slow
        # multiprocess CPU consumption
        self.subdomain = None
        # This contains id of possibly former agents who may have left the company but still contributed to lots of
        # comments in the data
        self.default_timezone = None

        if load:
            assert Path(cache_dir).exists() and Path(cache_dir).is_dir()
            file_hash = hash_files([self.path])
            self.cache_path = Path(cache_dir) / f'{file_hash}.augemented_agents.cache'

            self.load()
            timezones = [profile['time_zone'] for profile in self.users.values() if
                         profile is not None and profile['role'] in ['agent', 'admin']]
            self.default_timezone = Counter(timezones).most_common(1)[0][0]

            # if self.cache_path.exists():
            #     self.augmented_agent_ids = pickle.load(self.cache_path.open('rb'))
            # else:
            #     self.augmented_agent_ids = self.get_augmented_agent_ids()
            #     pickle.dump(self.augmented_agent_ids, self.cache_path.open('wb'))

            self.augmented_agent_ids = []

            self.length = len(self.tickets)
        else:
            self.length = len(ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='tickets/',
                                                     cache_size=self.cache_size, load_in_memory=False))

    def get_augmented_agent_ids(self):
        agent_ids_profiles = [profile['id'] for profile in self.users.values() if profile is not None and profile['role'] in ['agent', 'admin']]
        agent_ids_groups = list(set(itertools.chain(*[g['members'] for g in self.groups.values()])))
        author_ticket_counter = Counter()

        for audits in tqdm(self.audits.values(), desc=f'Augmenting agent ids for {self.path}'):
            author_ids = []
            if audits is None:
                continue
            for audit in audits:
                for event in audit['events']:
                    if event['type'] == 'Comment':
                        author_id = event['author_id']
                        author_ids.append(author_id)

            for author_id in set(author_ids):
                author_ticket_counter[author_id] += 1

        agent_ids_tickets = []
        for author_id, freq in author_ticket_counter.most_common():
            agent_ids_tickets.append(author_id)
            # Assume that if author id is found in more than 100 tickets, it's most likely an agent
            if freq < 100:
                break

        augmented_agent_ids = list(set(agent_ids_profiles + agent_ids_groups + agent_ids_tickets))
        return augmented_agent_ids

    def load(self):
        self.tickets = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='tickets/',
                                              cache_size=self.cache_size, load_in_memory=False)
        self.groups = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='groups/',
                                             cache_size=self.cache_size, load_in_memory=False)
        self.audits = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='audits/',
                                             cache_size=self.cache_size, load_in_memory=False)
        self.users = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='users/',
                                            cache_size=self.cache_size, load_in_memory=True)
        self.macros = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='macros/',
                                             cache_size=self.cache_size, load_in_memory=True)
        self.brands = ZipFolderJsonFilesDict(zipfile_path=self.path, folder_path_suffix='brands/',
                                             cache_size=self.cache_size, load_in_memory=True)

        # if len(self.groups) == 0:
        #     raise ValueError('The zipfile does not contain group memberships. Contact Abhi right away!')

        self.idx2ticketid = {idx: ticket_id for idx, ticket_id in enumerate(sorted(self.tickets))}
        for ticket_id in self.tickets:
            self.subdomain = self.tickets[ticket_id]['url'].split('.')[0][8:]
            break

    def __getitem__(self, idx):
        ticket_id = self.idx2ticketid[idx]
        ticket_dict = self.tickets[ticket_id]

        audits_list = self.audits.get(ticket_id, None)

        if audits_list is None:
            log.warning(f'No audits found for ticket ID: {ticket_id}')
            return None
        if ticket_dict is None:
            log.warning(f'No ticket found for ticket ID: {ticket_id}')
            return None

        ticket = self.construct_ticket(ticket_dict, audits_list)
        return ticket

    def __len__(self):
        return self.length

    def construct_ticket(self, ticket_dict, audits_list):
        # No audits, no ticket
        ticket_created_at = convert_time_format(ticket_dict['created_at'], self.ZENDESK_TIME_FORMAT)
        ticket_subject = ticket_dict['raw_subject']
        ticket_channel = ticket_dict['via']['channel']
        ticket_id = ticket_dict['id']
        group_id = ticket_dict['group_id']

        comments = self.extract_comments(ticket_dict, audits_list)

        # Sometimes the audits initial audits don't contain the customer name but subsequent audits do
        author_names = {comment.author.id: comment.author.name for comment in comments if comment.author.name is not None}
        for comment in comments:
            if comment.author.name is None and comment.author.id in author_names:
                comment.author.name = author_names[comment.author.id]

        for comment in comments:
            # No one answered and ticket got converted to email
            if ticket_channel == 'chat' and comment.text.startswith('Chat started:'):
                ticket_channel = 'email'

        ticket = Ticket(ticket_id=ticket_id,
                        group_id=group_id,
                        subject=ticket_subject,
                        timestamp=ticket_created_at,
                        comments=comments,
                        subdomain=self.subdomain,
                        channel=ticket_channel)

        ticket.raw_ticket = ticket_dict
        ticket.raw_audits = audits_list

        return ticket

    def extract_comments(self, ticket_dict, audit_list):
        comments = []
        group_id = ticket_dict['group_id']
        for audit in audit_list:
            # ignore all events which come from system
            if audit['via']['channel'] == 'system':
                continue

            extracted_comments = self.extract_comment(audit, ticket_dict, group_id)
            for comment in extracted_comments:
                group_id = comment.group_id
                comments.append(comment)

        return comments

    def extract_comment(self, audit, ticket_dict, group_id):
        comments = []
        for event in audit['events']:
            if (event['type'] == 'Create' or event['type'] == 'Change') and event['field_name'] == 'group_id':
                group_id = event['value']
            elif event['type'] == 'Comment':
                author_id = event['author_id']
                author_role, author_name, author_alias, author_email, signature = None, None, None, None, None
                author_timezone = None
                created_at = convert_time_format(audit['created_at'], self.ZENDESK_TIME_FORMAT)
                comment_type = 'public' if event['public'] else 'private'

                text = event.get('plain_body', event.get('body', event.get('html_body')))
                text = html.unescape(text)

                language = None

                # Added None protection is the json was not read due to some decoding fuck up
                if author_id in self.users and self.users[author_id] is not None:
                    author_role = self.users[author_id]['role']
                    author_name = self.users[author_id]['name']
                    author_alias = self.users[author_id]['alias']
                    author_email = self.users[author_id]['email']
                    author_timezone = self.users[author_id]['time_zone']
                    signature = self.get_agent_signatures(author_id)
                elif author_id in self.augmented_agent_ids:
                    author_role = 'agent'
                    author_timezone = self.default_timezone
                else:
                    author_role = 'end-user'


                # if author_role is None:
                #     # assumes that only agents can make private comment
                #     if comment_type == 'private':
                #         author_role = 'agent'
                #     # assumes ticket requester is typically user
                #     elif ticket_dict['requester_id'] == author_id:
                #         author_role = 'end-user'
                #     else:
                #         author_role = 'end-user'

                try:
                    # We only need customer first name so it is better to assume the name to be end-user
                    if author_name is None and author_alias is None and author_role == 'end-user':
                        author_name = audit['via']['source']['from'].get('name', None)
                        author_email = audit['via']['source']['from'].get('email', None)
                    if author_name is None and author_role == 'end-user':
                        author_name = ticket_dict['via']['source']['from'].get('name', None)
                        if author_email is None:
                            author_email = ticket_dict['via']['source']['from'].get('email', None)

                except KeyError as e:
                    pass

                macros = []

                if self.macro_extract:
                    macros = self.extract_macros(audit)

                author = Author(author_id=author_id,
                                role=author_role,
                                name=author_name,
                                email=author_email,
                                alias=author_alias,
                                signature=signature,
                                timezone=author_timezone
                                )
                comment = Comment(text=text,
                                  author=author,
                                  created_timestamp=created_at,
                                  language=language,
                                  macros=macros,
                                  comment_type=comment_type,
                                  group_id=group_id)
                comments.append(comment)
        return comments

    def extract_macros(self, audit):
        macros = []
        for e in audit['events']:
            if e['type'] == 'AgentMacroReference':
                macro_id = int(e['macro_id'])
                if macro_id in self.macros:
                    macro = self.macros[macro_id]
                    title = macro['title']
                    restriction = macro['restriction']['type'] if macro['restriction'] is not None else None
                    active = macro['active']
                    macros.append(Macro(macro_id=macro_id,
                                        title=title,
                                        active=active,
                                        restriction_type=restriction))
        return macros

    def get_agent_signatures(self, author_id):
        if author_id not in self.users:
            return None

        user_dict = self.users[author_id]
        url = user_dict['url']
        subdomain = url[8:url.index('.')]

        brands = [b for b in self.brands.values() if b['subdomain'] == subdomain]

        if len(brands) > 0:
            brand = brands[0]
            brand_signature_template = brand['signature_template']
        else:
            brand_signature_template = '{{agent.' + 'signature' + '}}'

        # Agent signature can contain markdown
        # https://support.zendesk.com/hc/en-us/articles/203691016-Formatting-text-with-Markdown

        # And map of available agent fields
        # https://support.zendesk.com/hc/en-us/articles/218869638-Using-Liquid-markup-to-set-agent-signatures
        # TODO(abhi) I am leaving out {{agent.organization.name}} as it is tricky to get it right away
        available_fields = [
            'name', 'first_name', 'last_name', 'role', 'email', 'phone',
            'signature'
        ]

        if brand_signature_template is None:
            brand_signature_template = '{{agent.' + 'signature' + '}}'

        signature = brand_signature_template
        for field in available_fields:
            template = '{{agent.' + field + '}}'
            if template in signature:
                if field == 'first_name':
                    value = user_dict['name'].split(' ')[0]
                elif field == 'last_name':
                    value = user_dict['name'].split(' ')[-1]
                elif field == 'signature':
                    value = markdown_to_text(user_dict[field])
                else:
                    value = user_dict[field]

                value = '' if value is None else value

                signature = signature.replace(template, value)
        return signature

    def __del__(self):
        del self.tickets
        del self.audits
        del self.brands
        del self.macros
        del self.users


class ZendeskDatasetTicketGen:
    def __init__(self, path, num_workers=0, indices=None, cache_dir='/tmp'):
        self.path = path
        self.dataset = ZippedZendeskDataset(path=self.path, load=True, cache_dir=cache_dir)
        self.indices = indices
        self.dataloader = DatasetIterator(dataset=self.dataset, num_workers=num_workers, indices=self.indices)
        self.num_invalid_tickets = 0

    def __iter__(self):
        for ticket in self.dataloader:
            if ticket is not None:
                yield ticket
            else:
                self.num_invalid_tickets += 1

    def __len__(self):
        return len(self.dataset) - self.num_invalid_tickets

    def __del__(self):
        del self.dataloader
        del self.dataset


class ZendeskTicketGen:
    def __init__(self, paths, num_workers=0, indices: list = None, verbose=False, detect_language=False,
                 max_num_tickets_per_dataset: int = None, cache_dir='/tmp'):
        """
        :param paths: path or paths to zip files
        :param num_workers: number of parallel workers, 0 to disable parallelization
        :param indices: list of indices to iterate over
        :param verbose: whether to be extra verbose
        :param max_num_tickets_per_dataset: if specified, randomly chooses these many tickets per dataset
        """
        self.paths = [paths] if not isinstance(paths, list) else paths
        self.indices = []
        self.cache_dir = cache_dir
        self.detect_language = detect_language
        self.num_workers = num_workers
        self.verbose = verbose
        prev_dataset_len = 0

        if max_num_tickets_per_dataset is not None and indices is not None:
            raise ValueError('Both max_num_tickets_per_dataset and indices cannot be specified')

        for path in tqdm(self.paths, desc='Parsing datasets'):
            num_tickets = len(ZippedZendeskDataset(path=path, load=False, cache_dir=self.cache_dir))
            if indices is not None:
                start_idx, end_idx = prev_dataset_len, prev_dataset_len + num_tickets
                indices_path = [idx - start_idx for idx in indices if end_idx > idx >= start_idx]
            elif max_num_tickets_per_dataset is not None:
                indices_path = list(range(num_tickets))
                np.random.shuffle(indices_path)
                indices_path = indices_path[:max_num_tickets_per_dataset]
            else:
                indices_path = list(range(num_tickets))
            self.indices.append(indices_path)
            if self.verbose:
                log.info(f'Loading {path} with {num_tickets} tickets')
            prev_dataset_len = num_tickets

        self.total_num_tickets = sum(len(indices) for indices in self.indices)

    def __iter__(self):
        for idx, path in enumerate(self.paths):
            ticket_gen = ZendeskDatasetTicketGen(path=path, num_workers=self.num_workers,
                                                 indices=self.indices[idx],
                                                 cache_dir=self.cache_dir)
            if self.verbose:
                log.info('Starting iterating over {}'.format(ticket_gen.path))
            for ticket in ticket_gen:
                yield ticket
            del ticket_gen
            gc.collect()

    def __len__(self):
        return self.total_num_tickets


def extract_agent_customer_names(comments):
    customer_name, agent_name = None, None
    for comment in comments[::-1]:
        name = comment.author.name
        name = name if name is not None and len(name) > 1 else None

        if comment.author.role == 'end-user':
            customer_name = name if customer_name is None else customer_name
        else:
            agent_name = name if agent_name is None else agent_name

    return customer_name, agent_name


def replace_name_by_token(text, name, name_token, first_name_token):
    if name is not None and len(name) > 0:
        customer_first_name = name.split(' ')[0]
        for token, c_name in [(name_token, name),
                              (first_name_token, customer_first_name)]:
            if isinstance(c_name, str):
                pattern = re.compile(re.escape(c_name), re.IGNORECASE)
                text = pattern.sub(token, text)

    return text


def detect_name_from_greeting(text):
    # Detect customer name
    customer_name = None
    first_line_split = text.split('\n')[0].split(',')[0].split('!')[0].strip().split()
    if len(first_line_split) == 2:
        customer_name = first_line_split[-1]
    elif len(first_line_split) == 3:
        customer_name = ' '.join(first_line_split[1:])

    if customer_name is not None and len(customer_name) < 3:
        customer_name = None

    return customer_name


def replace_signature(text, replace_by, signature_counter=None):
    if signature_counter is None:
        return text

    for signature, _ in signature_counter.most_common():
            matched_signature = find_matching_signature(text, signature)
            if matched_signature is not None:
                text = text.replace(matched_signature, replace_by)
                break
    return text


def contruct_query(text, customer_name, agent_name, signature_counter=None, max_response_len=None, add_eod_token=True):
    detected_customer_name = detect_name_from_greeting(text)

    comment_text = replace_signature(text, replace_by=SIGNATURE, signature_counter=signature_counter)

    comment_text = replace_name_by_token(text=comment_text,
                                         name=customer_name,
                                         name_token=CUSTOMER_NAME,
                                         first_name_token=CUSTOMER_FNAME)

    if detected_customer_name is not None and CUSTOMER_FNAME not in comment_text:
        comment_text = replace_name_by_token(text=comment_text,
                                             name=detected_customer_name,
                                             name_token=CUSTOMER_NAME,
                                             first_name_token=CUSTOMER_FNAME)

    comment_text = replace_name_by_token(text=comment_text,
                                         name=agent_name,
                                         name_token=AGENT_NAME,
                                         first_name_token=AGENT_FNAME)

    comment_text = comment_text[:max_response_len] if max_response_len is not None else comment_text

    return ' '.join([BOD, comment_text, EOD]) if add_eod_token else ' '.join([BOD, comment_text])


