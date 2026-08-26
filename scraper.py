import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
REQUEST_DELAY = 0.5

session = requests.Session()
session.headers.update(HEADERS)


def extract_meta(soup, *keys):
    for key in keys:
        for attr in ['name', 'property']:
            meta = soup.find('meta', attrs={attr: key})
            if meta and meta.get('content'):
                return meta['content'].strip()
    return ''


KATHIMERINI_SECTION_MAP = {
    'society': 'Ελλάδα / Κοινωνία',
    'politics': 'Πολιτική',
    'world': 'Διεθνή',
    'culture': 'Τέχνες / Πολιτισμός',
    'economy': 'Οικονομία',
    'opinions': 'Απόψεις',
    'technology': 'Τεχνολογία',
    'sports': 'Αθλητισμός',
}


def extract_author_kathimerini(soup):
    text = soup.get_text()
    m = re.search(r'(?:Newsroom|Ιωάννα Μάνδρου)', text)
    if m:
        return m.group(0)
    return 'Newsroom'


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def scrape_kathimerini(url):
    r = session.get(url, timeout=20)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.entry-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = ''
    path_parts = urlparse(url).path.strip('/').split('/')
    if path_parts:
        raw = path_parts[0].lower()
        section = KATHIMERINI_SECTION_MAP.get(raw, raw.replace('-', ' / ').title())

    author = extract_author_kathimerini(soup)
    website = 'kathimerini.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_efsyn(url):
    r = session.get(url, timeout=20)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.article-body')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')

    author_el = soup.select_one('.field-name-field-author .field-item, .author, .byline')
    author = clean_text(author_el.get_text()) if author_el else 'Newsroom'
    if not author or author.lower() in ('efsyn.gr', ''):
        author = 'Newsroom'
    website = 'efsyn.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_skai(url):
    r = session.get(url, timeout=20)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.post-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')
    author = extract_meta(soup, 'article:author')
    if not author:
        author = 'Newsroom'
    website = 'skai.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_zougla(url):
    r = session.get(url, timeout=20)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.entry-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')

    text = soup.get_text()
    author_match = re.search(r'([Α-ΩΆ-Ώ][α-ωά-ώ]+\s[Α-ΩΆ-Ώ][α-ωά-ώ]+)\s+\d{2}\.\d{2}\.\d{4}', text)
    author = clean_text(author_match.group(1)) if author_match else 'Newsroom'
    website = 'zougla.gr'

    return url, title, body_text, author, website, pub_date, section


SCRAPERS = {
    'kathimerini.gr': scrape_kathimerini,
    'efsyn.gr': scrape_efsyn,
    'skai.gr': scrape_skai,
    'zougla.gr': scrape_zougla,
}


def get_scraper(url):
    domain = urlparse(url).netloc.lower()
    for key in SCRAPERS:
        if key in domain:
            return SCRAPERS[key]
    return None


def main():
    input_file = 'links.txt'
    output_file = 'sample_database.csv'

    urls = []
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(urls)} URLs from {input_file}")

    rows = []
    total = len(urls)
    for i, url in enumerate(urls, 1):
        scraper = get_scraper(url)
        if not scraper:
            print(f"[{i}/{total}] ⚠️ No scraper for {url}")
            continue

        try:
            result = scraper(url)
            rows.append(result)
            print(f"[{i}/{total}] ✅ {urlparse(url).netloc} - {result[1][:50]}")
        except Exception as e:
            print(f"[{i}/{total}] ❌ {url}: {e}")

        time.sleep(REQUEST_DELAY)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'title', 'full-text', 'author', 'website', 'datetime', 'section'])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} articles to {output_file}")


if __name__ == '__main__':
    main()
