import html
import math
from datetime import timedelta

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from markdown import markdown

import logging


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    if markdown_string is None:
        return None

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    # apparently it leaves the []
    text = text.replace('[]', '')
    return text


def convert_time_format(input_time_string, time_format=None):
    datetime_object = pd.to_datetime(input_time_string, infer_datetime_format=True, format=time_format)
    return datetime_object.isoformat()

zone_mapping = {}
zone_mapping["International Date Line West"] = "-11:00"
zone_mapping["Midway Island"] = "-11:00"
zone_mapping["Hawaii"] = "-10:00"
zone_mapping["Alaska"] = "-08:00"
zone_mapping["American Samoa"] = "-11:00"
zone_mapping["Arizona"] = "-07:00"
zone_mapping["Pacific Time (US &amp; Canada)"] = "-07:00"
zone_mapping["Tijuana"] = "-07:00"
zone_mapping["Central America"] = "-06:00"
zone_mapping["Chihuahua"] = "-06:00"
zone_mapping["Mazatlan"] = "-06:00"
zone_mapping["Mountain Time (US &amp; Canada)"] = "-06:00"
zone_mapping["Saskatchewan"] = "-06:00"
zone_mapping["Bogota"] = "-05:00"
zone_mapping["Central Time (US &amp; Canada)"] = "-05:00"
zone_mapping["Guadalajara"] = "-05:00"
zone_mapping["Lima"] = "-05:00"
zone_mapping["Mexico City"] = "-05:00"
zone_mapping["Monterrey"] = "-05:00"
zone_mapping["Quito"] = "-05:00"
zone_mapping["Caracas"] = "-04:00"
zone_mapping["Eastern Time (US &amp; Canada)"] = "-04:00"
zone_mapping["Georgetown"] = "-04:00"
zone_mapping["Indiana (East)"] = "-04:00"
zone_mapping["La Paz"] = "-04:00"
zone_mapping["Puerto Rico"] = "-04:00"
zone_mapping["Santiago"] = "-04:00"
zone_mapping["Atlantic Time (Canada)"] = "-03:00"
zone_mapping["Brasilia"] = "-03:00"
zone_mapping["Buenos Aires"] = "-03:00"
zone_mapping["Montevideo"] = "-03:00"
zone_mapping["Newfoundland"] = "-02:30"
zone_mapping["Greenland"] = "-02:00"
zone_mapping["Mid-Atlantic"] = "-02:00"
zone_mapping["Cape Verde Is."] = "-01:00"
zone_mapping["Azores"] = "+00:00"
zone_mapping["Monrovia"] = "+00:00"
zone_mapping["UTC"] = "+00:00"
zone_mapping["Casablanca"] = "+01:00"
zone_mapping["Dublin"] = "+01:00"
zone_mapping["Edinburgh"] = "+01:00"
zone_mapping["Lisbon"] = "+01:00"
zone_mapping["London"] = "+01:00"
zone_mapping["West Central Africa"] = "+01:00"
zone_mapping["Amsterdam"] = "+02:00"
zone_mapping["Belgrade"] = "+02:00"
zone_mapping["Berlin"] = "+02:00"
zone_mapping["Bern"] = "+02:00"
zone_mapping["Bratislava"] = "+02:00"
zone_mapping["Brussels"] = "+02:00"
zone_mapping["Budapest"] = "+02:00"
zone_mapping["Cairo"] = "+02:00"
zone_mapping["Copenhagen"] = "+02:00"
zone_mapping["Harare"] = "+02:00"
zone_mapping["Kaliningrad"] = "+02:00"
zone_mapping["Ljubljana"] = "+02:00"
zone_mapping["Madrid"] = "+02:00"
zone_mapping["Oslo"] = "+02:00"
zone_mapping["Paris"] = "+02:00"
zone_mapping["Prague"] = "+02:00"
zone_mapping["Pretoria"] = "+02:00"
zone_mapping["Rome"] = "+02:00"
zone_mapping["Sarajevo"] = "+02:00"
zone_mapping["Skopje"] = "+02:00"
zone_mapping["Stockholm"] = "+02:00"
zone_mapping["Vienna"] = "+02:00"
zone_mapping["Warsaw"] = "+02:00"
zone_mapping["Zagreb"] = "+02:00"
zone_mapping["Athens"] = "+03:00"
zone_mapping["Baghdad"] = "+03:00"
zone_mapping["Bucharest"] = "+03:00"
zone_mapping["Helsinki"] = "+03:00"
zone_mapping["Istanbul"] = "+03:00"
zone_mapping["Jerusalem"] = "+03:00"
zone_mapping["Kuwait"] = "+03:00"
zone_mapping["Kyev"] = "+03:00"
zone_mapping["Kyiv"] = "+03:00"
zone_mapping["Minsk"] = "+03:00"
zone_mapping["Moscow"] = "+03:00"
zone_mapping["Nairobi"] = "+03:00"
zone_mapping["Riga"] = "+03:00"
zone_mapping["Riyadh"] = "+03:00"
zone_mapping["Sofia"] = "+03:00"
zone_mapping["St. Petersburg"] = "+03:00"
zone_mapping["Tallinn"] = "+03:00"
zone_mapping["Vilnius"] = "+03:00"
zone_mapping["Abu Dhabi"] = "+04:00"
zone_mapping["Baku"] = "+04:00"
zone_mapping["Muscat"] = "+04:00"
zone_mapping["Samara"] = "+04:00"
zone_mapping["Tbilisi"] = "+04:00"
zone_mapping["Volgograd"] = "+04:00"
zone_mapping["Yerevan"] = "+04:00"
zone_mapping["Kabul"] = "+04:30"
zone_mapping["Tehran"] = "+04:30"
zone_mapping["Ekaterinburg"] = "+05:00"
zone_mapping["Islamabad"] = "+05:00"
zone_mapping["Karachi"] = "+05:00"
zone_mapping["Tashkent"] = "+05:00"
zone_mapping["Chennai"] = "+05:30"
zone_mapping["Kolkata"] = "+05:30"
zone_mapping["Mumbai"] = "+05:30"
zone_mapping["New Delhi"] = "+05:30"
zone_mapping["Sri Jayawardenepura"] = "+05:30"
zone_mapping["Kathmandu"] = "+05:45"
zone_mapping["Almaty"] = "+06:00"
zone_mapping["Astana"] = "+06:00"
zone_mapping["Dhaka"] = "+06:00"
zone_mapping["Urumqi"] = "+06:00"
zone_mapping["Rangoon"] = "+06:30"
zone_mapping["Bangkok"] = "+07:00"
zone_mapping["Hanoi"] = "+07:00"
zone_mapping["Jakarta"] = "+07:00"
zone_mapping["Krasnoyarsk"] = "+07:00"
zone_mapping["Novosibirsk"] = "+07:00"
zone_mapping["Beijing"] = "+08:00"
zone_mapping["Chongqing"] = "+08:00"
zone_mapping["Hong Kong"] = "+08:00"
zone_mapping["Irkutsk"] = "+08:00"
zone_mapping["Kuala Lumpur"] = "+08:00"
zone_mapping["Perth"] = "+08:00"
zone_mapping["Singapore"] = "+08:00"
zone_mapping["Taipei"] = "+08:00"
zone_mapping["Ulaan Bataar"] = "+08:00"
zone_mapping["Ulaanbaatar"] = "+08:00"
zone_mapping["Osaka"] = "+09:00"
zone_mapping["Sapporo"] = "+09:00"
zone_mapping["Seoul"] = "+09:00"
zone_mapping["Tokyo"] = "+09:00"
zone_mapping["Yakutsk"] = "+09:00"
zone_mapping["Adelaide"] = "+09:30"
zone_mapping["Darwin"] = "+09:30"
zone_mapping["Brisbane"] = "+10:00"
zone_mapping["Canberra"] = "+10:00"
zone_mapping["Guam"] = "+10:00"
zone_mapping["Hobart"] = "+10:00"
zone_mapping["Melbourne"] = "+10:00"
zone_mapping["Port Moresby"] = "+10:00"
zone_mapping["Sydney"] = "+10:00"
zone_mapping["Vladivostok"] = "+10:00"
zone_mapping["Magadan"] = "+11:00"
zone_mapping["New Caledonia"] = "+11:00"
zone_mapping["Solomon Is."] = "+11:00"
zone_mapping["Srednekolymsk"] = "+11:00"
zone_mapping["Auckland"] = "+12:00"
zone_mapping["Fiji"] = "+12:00"
zone_mapping["Kamchatka"] = "+12:00"
zone_mapping["Marshall Is."] = "+12:00"
zone_mapping["Wellington"] = "+12:00"
zone_mapping["Chatham Is."] = "+12:45"
zone_mapping["Nuku'alofa"] = "+13:00"
zone_mapping["Samoa"] = "+13:00"
zone_mapping["Tokelau Is."] = "+13:00"

ZONE_MAPPING = {html.unescape(k): v for k, v in zone_mapping.items()}


def encoder_periodic_value(val, period):
    x = np.sin(2 * math.pi * val / period)
    y = np.cos(2 * math.pi * val / period)
    return x, y


def time_encoder(timestamp, timezone=None):
    if timezone is None or timezone not in ZONE_MAPPING:
        logging.warning(f'timezone {timezone} is not recognized. Falling back to UTC')
        timezone = "UTC"

    time_delta = ZONE_MAPPING[timezone]
    multiplier = -1 if time_delta[0] == '-' else 1
    hours = int(time_delta[1:3])
    minutes = int(time_delta[4:6])
    datetime_local = pd.to_datetime(timestamp) + multiplier * timedelta(hours=hours, minutes=minutes)
    datetime_local = pd.to_datetime(datetime_local)

    feature = [*encoder_periodic_value(datetime_local.dayofweek, 7.0),
               *encoder_periodic_value(datetime_local.dayofyear, 365.0),
               *encoder_periodic_value(datetime_local.weekofyear, 52.0),
               *encoder_periodic_value(datetime_local.daysinmonth, datetime_local.days_in_month),
               *encoder_periodic_value(datetime_local.hour, 24)]
    return feature
