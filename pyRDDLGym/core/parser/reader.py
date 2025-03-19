import re

from pyRDDLGym.core.debug.exception import RDDLParseError

REPLACEMENT_CHAR_BYTES = b"\xef\xbf\xbd"


class RDDLReader(object):
    comment = r"\/\/.*?\n"
    comment_ws = r"(\s*?\n\s*?)+"
    domain_block = r"(?s)domain.*?\{.*?pvariables.*?cpfs.*?reward.*?;.*?\}[^;]"
    nonfluent_in_domain = r"(?s)\{\s*?non-fluent,.*?\};"
    nonfluent_block = r"(?s)non-fluents[^=]*?\{.*?\}[^;]"
    instance_block = r"(?s)instance.*?\{.*\}[^;]"

    def __init__(self, dom, inst=None):
        with open(dom, encoding="utf-8", errors="replace") as file:
            dom_txt = file.read()
        dom_txt = self._remove_comments(dom_txt)
        dom_txt = dom_txt + "\n"

        if inst is not None:
            with open(inst, encoding="utf-8", errors="replace") as file:
                inst_txt = file.read()
            inst_txt = self._remove_comments(inst_txt)
            dom_txt = dom_txt + "\n\n" + inst_txt + "\n"

        # check for decoding errors
        byte_data = dom_txt.encode("utf-8")
        if REPLACEMENT_CHAR_BYTES in byte_data:
            raise RDDLParseError(
                (
                    "UnicodeDecodeError: Invalid byte sequence encountered",
                    "in file after removing comments.",
                )
            )

        # inspect rddl if three block are present - domain, non-fluent, instance
        m = re.search(self.domain_block, dom_txt)
        if m is None:
            if 'reward' not in dom_txt:
                raise RDDLParseError(
                    "reward expression is missing or contains a syntax error."
                )
            elif 'cpfs' not in dom_txt:
                raise RDDLParseError(
                    "cpfs {...} block is missing or contains a syntax error."
                )
            elif 'pvariables' not in dom_txt:
                raise RDDLParseError(
                    "pvariables {...} block is missing or contains a syntax error."
                )
            else:
                raise RDDLParseError(
                    "domain {...} block is missing or contains a syntax error."
                )

        domaintxt = m.group(0)
        m = re.search(self.nonfluent_in_domain, domaintxt)
        if m is not None:
            m = re.search(self.nonfluent_block, dom_txt)
            if m is None:
                raise RDDLParseError(
                    "non-fluents {...} block is missing or contains a syntax error."
                )

        m = re.search(self.instance_block, dom_txt)
        if m is None:
            raise RDDLParseError(
                "instance {...} block is missing or contains a syntax error."
            )

        self.dom_txt = dom_txt

    @property
    def rddltxt(self):
        return self.dom_txt

    def _remove_comments(self, txt):
        txt = re.sub(self.comment, "\n", txt)
        txt = re.sub(self.comment_ws, "\n", txt)
        return txt
