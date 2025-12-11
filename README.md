# content-overlap-check
Developing a tool to check for content overlap between an inputted text and published topics.

## Purpose

This code will be designed to be a tool for crosschecking between different proposals for collections.
We want to minimise the overlap and make sure that the overlap in content is kept to a minimum.


## Usage

1. Install dependencies (if not already installed):

```bash
pip install requests beautifulsoup4
```

2. Run the checker with a proposal text file (default topics URL):

```bash
python3 c-over-checker.py new_proposal.txt
```

3. Run the checker and specify a custom topics page URL:

```bash
python3 c-over-checker.py new_proposal.txt --topics-url "https://example.org/topics"
```

The script will fetch the main topics page, follow links to individual topic pages when possible, and report any detected overlaps between the proposal text and existing topic descriptions.

## Notes

- Dependencies are listed in `requirements.txt`; install with:

```bash
pip install -r requirements.txt
```

- `proposal` accepts a local file path, an `http(s)` URL (page text will be extracted), or `-` to read from stdin. Examples:

```bash
# local file
python3 c-over-checker.py new_proposal.txt

# proposal from a web page
python3 c-over-checker.py https://example.org/proposal-page

# read proposal from stdin
cat new_proposal.txt | python3 c-over-checker.py -
```

