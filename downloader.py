import os.path

import httpx
from bs4 import BeautifulSoup


def get_available_files(url):
    resp = httpx.get(url)
    # print(resp.json())
    # obj = resp.json()
    # for f in obj['files']:
    #     print(f)
    #     dpath = f['downloadUri']
    #
    #     download_and_save_file(dpath, name=f['name'], output_dir='out')

    # soup = BeautifulSoup(resp.text, 'html.parser')
    # for link in soup.find_all('a'):
    #     # print(link)
    #     href = link.get('href')
    #     if href is not None:
    #         if href.startswith('/files'):
    #             print(link.get('href'))

    # for link in soup.find_all('span'):
    #     print(link)
    #     nested_element = link.find_all('span', id='sb-download-all-link')
    #     print(nested_element)
    #     dataurl = link.get('id')
    #     if dataurl is not None:
    #         # print(dataurl)
    #         if dataurl.startswith('/catalog/file'):
    #             print(link.get('class'))


def download_and_save_file(url, name=None, extension=None, output_dir=None):

    if name is None:
        name = url.split('/')[-1]

    if output_dir:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        name = os.path.join(output_dir, name)

    if extension is not None:
        name = f'{name}{extension}'

    resp = httpx.get(url, timeout=60)
    if resp.status_code == 200:
        with open(name, 'wb') as wfile:
            wfile.write(resp.content)
    else:
        print(f'failed getting data from {url}. status={resp.status_code}. text={resp.text}')


def main():
    x = ((37 ** 2) + (27.5 ** 2))**(1/2)
    print(x)

    url = 'https://gdr.openei.org/submissions/954'
    url = 'https://www.sciencebase.gov/catalog/item/63a20e8cd34e176674f51d51'
    url = 'https://www.sciencebase.gov/catalog/item/download/63a20e8cd34e176674f51d51?format=json'
    url = 'https://www.sciencebase.gov/catalog/file/get/63a20e8cd34e176674f51d51'
    # get_available_files(url)

    # url = 'https://gdr.openei.org/files/954/Tularosa_170511_mdl13_cellcenter.dat'
    # download_and_save_file(url, extension='.zip', output_dir='out')


if __name__ == '__main__':
    main()
