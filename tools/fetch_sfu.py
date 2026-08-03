"""Download the SFU R reference implementations into tools/sfu/. Needs network, not R."""

import os
import re
import urllib.request

SLUGS = """
ackleyr bealer boha1r boha2r boha3r boothr braninmodif braninr braninsc bukin6r camel3r camel6r
colviller crossitr dejong5r dixonprr dropr easomr eggr forretal08lc forretal08r goldprr goldprsc
griewankr grlee12r hart3r hart4r hart6r hart6sc holderr langerr levy13r levyr matyar mccormr michalr
perm0dbr permdbr powellr powersumr rastrr rosenr rosensc rothypr schaffer2r schaffer4r schwefr
shekelr shubertr spherefmod spherefr stybtangr sumpowr sumsqur tridr zakharovr
""".split()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfu")


def fetch(slug: str) -> str:
    url = f"https://www.sfu.ca/~ssurjano/Code/{slug}.html"
    with urllib.request.urlopen(url) as response:
        html = response.read().decode("utf-8", "replace")
    body = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.S)
    assert body, f"no <pre> block at {url}"
    return re.sub(
        r"&lt;|&gt;|&amp;",
        lambda m: {"&lt;": "<", "&gt;": ">", "&amp;": "&"}[m[0]],
        body[1],
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for slug in SLUGS:
        with open(f"{OUT}/{slug}.R", "w") as fp:
            fp.write(fetch(slug))
    print(f"wrote {len(SLUGS)} R sources to {OUT}")
