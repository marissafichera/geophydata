import os.path
import sys

import httpx
from bs4 import BeautifulSoup


def get_available_files(url):
    resp = httpx.get(url)

    soup = BeautifulSoup(resp.text, 'html.parser')
    for link in soup.find_all('a'):
        # print(link)
        href = link.get('href')
        if href is not None:
            if href.startswith('/files'):
                print(link.get('href'))


def download_and_save_file(url, name=None, extension=None, output_dir=None, gis_id=None):

    if name is None:
        name = url.split('/')[-1]

    if os.path.join(output_dir, gis_id):
        if not os.path.isdir(os.path.join(output_dir, gis_id)):
            os.mkdir(os.path.join(output_dir, gis_id))

        name = os.path.join(output_dir, gis_id, name)

    if extension is not None:
        name = f'{name}{extension}'

    resp = httpx.get(url, timeout=60)
    if resp.status_code == 200:
        with open(name, 'wb') as wfile:
            wfile.write(resp.content)
    else:
        print(f'failed getting data from {url}. status={resp.status_code}. text={resp.text}')


def main():
    gis_dir = r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData'
    url = 'https://www.sciencebase.gov/catalog/item/5f2978e682cef313ed9e82aa'

    if url.startswith('https://gdr'):
        get_available_files(url)
        download_and_save_file(url, extension=None, output_dir=gis_dir, gis_id=None)

    if url.startswith('https://www.sciencebase.gov'):
        base = url.split('/')[0:-2]
        base_list = []
        for b in base:
            base_list.append(f'{b}')

        catid = url.split('/')[-1]

        dataurl_list = ['file', 'get', f'{catid}']

        d_elements = base_list + dataurl_list
        dataurl = '/'.join(d_elements)
        download_and_save_file(dataurl, extension='.zip', output_dir=gis_dir, gis_id='USGS015')

    # url = 'https://www.sciencebase.gov/catalog/file/get/63a20e8cd34e176674f51d51'
    # get_available_files(url)

    # url = 'https://gdr.openei.org/files/954/Tularosa_170511_mdl13_cellcenter.dat'
    # download_and_save_file(url, extension='.zip', output_dir='out')


if __name__ == '__main__':
    main()
