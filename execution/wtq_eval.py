import os
import sqlite3
import json
import random

from typing import List, Dict, Any
from collections import defaultdict

import sys, os, re, argparse
import unicodedata
from codecs import open
from math import isnan, isinf
from abc import ABCMeta, abstractmethod

DB_DIR = "data/squall/tables/db"

UNITS = ["km", "years", "months", "days", "kg", "changes", "min", "miles", "points", "kW"]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

################ String Normalization ################

def normalize(x):
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' +  str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)

def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################

def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.
    
    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


################ Batch Mode ################

def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')

def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]

def wtq_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> Any:
    # sql = "SELECT " + sql
    
    db_id = example["db_id"]
    db_path = os.path.join(DB_DIR, db_id + ".db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        return cursor.fetchall()
    # except sqlite3.OperationalError as e:
    #     error_msg = f"ERROR: {str(e)}"
    #     print(f"Error {str(e)} in execution sql query {sql}")
    #     if return_error_msg:
    #         return error_msg
    #     else:
    #         return None
    except:
        return None

def wtq_answer_eq(prediction: List[Any], gold_answer: str, allow_normalize: bool = True) -> bool:
    if prediction is None or len(prediction) == 0: 
        return False
    
    gold_answer_str = gold_answer
    if "|" in gold_answer:
        gold_answer = gold_answer.split("|")
    else:
        gold_answer = [gold_answer]

    if len(prediction) != 1:
        # if not all([len(s) == 1 for s in prediction]):
        #     print(f"warning: prediction has multiple col per row: {prediction}, but gold answer: {gold_answer}")
        # assert all([len(s) == 1 for s in prediction])
        prediction = [(s[0] for s in prediction)]

    pred_vals = to_value_list([str(s) for s in prediction[0]])
    gold_vals = to_value_list(gold_answer)

    eval_result = check_denotation(pred_vals, gold_vals)

    # give another try by normalize out the units
    if not eval_result and allow_normalize:
        num_unit_split = gold_answer_str.split(" ")
        if "," in gold_answer_str and is_number(gold_answer_str.replace(",", "")):
            second_chance_eval_result = wtq_answer_eq(prediction, gold_answer_str.replace(",", ""))
            # print(f"gold answer {gold_answer_str} and prediction {prediction}")
        elif len(num_unit_split) == 2 and num_unit_split[1] in UNITS and is_number(num_unit_split[0]):
            second_chance_eval_result = wtq_answer_eq(prediction, num_unit_split[0])
        elif gold_answer_str[0] in ["$", "£"]:
                second_chance_eval_result = wtq_answer_eq(prediction, gold_answer_str[1:])
        elif gold_answer_str.endswith("th") and is_number(gold_answer_str[:-2]):
            second_chance_eval_result = wtq_answer_eq(prediction, gold_answer_str[:-2])
        else:
            second_chance_eval_result = False

        # if second_chance_eval_result:
        #     print(f"second chance win: original answer: {gold_answer_str}, second chance answer: {num_unit_split[0]}, prediction: {prediction}")
        
        return second_chance_eval_result
    
    return eval_result


