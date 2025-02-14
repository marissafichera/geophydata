import os.path
import sys

import httpx
from bs4 import BeautifulSoup
import requests


def get_available_files(url, gis_dir, gis_id):
    resp = httpx.get(url)
    # Split the URL by slashes and exclude the last segment
    parts = url.rstrip('/').split('/')

    # Remove the last segment (the resource or endpoint)
    base_url = '/'.join(parts[:-1])

    # List of allowed extensions
    allowed_extensions = ['.gz', '.tif', '.tfw', '.zip']

    soup = BeautifulSoup(resp.text, 'html.parser')

    #for GDR data
    # for link in soup.find_all('a'):
    #     href = link.get('href')
    #     if href is not None:
    #         if href.startswith('/files'):
    #             fileurl = link.get('href')
    #             download_and_save_file(url, name=fileurl, output_dir=gis_dir, gis_id=gis_id)

    # for USGS data
    # Find all <a> tags inside <td> elements
    links = soup.find_all('td', {'class': None})  # Or add a class if needed
    # print(f'{links=}')
    # Extract the href attribute and link text
    for td in links:
        # print(f'{td=}')
        link = td.find('a', href=True)
        print(f'{link=}')
        if link is not None:
            print(f"Link text: {link.text}, URL: {link['href']}")
            if any(link['href'].endswith(ext) for ext in allowed_extensions):  # Check for valid extension
                fileurl = link.get('href')
                print(f'{fileurl=}')
                url = base_url + fileurl
                download_and_save_file(url, output_dir=gis_dir, gis_id=gis_id)


def download_and_save_file(url, name=None, extension=None, output_dir=None, gis_id=None):
    if name is None:
        name = url.split('/')[-1]

    if os.path.join(output_dir, gis_id):
        if not os.path.isdir(os.path.join(output_dir, gis_id)):
            os.mkdir(os.path.join(output_dir, gis_id))

    if extension is not None:
        name = f'{name}{extension}'

    print(f'{url}')
    print(name)
    resp = httpx.get(url, timeout=60)
    if resp.status_code == 200:
        with open(name, 'wb') as wfile:
            wfile.write(resp.content)
    else:
        print(f'failed getting data from {url}. status={resp.status_code}. text={resp.text}')


def main():
    gis_dir = r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData'
    url = 'https://mrdata.usgs.gov/gravity/'

    if url.startswith('https://gdr'):
        get_available_files(url, gis_dir)
        # download_and_save_file(url, extension=None, output_dir=gis_dir, gis_id='FORGE001')

    if url.startswith('https://www.sciencebase.gov'):
        base = url.split('/')[0:-2]
        base_list = []
        for b in base:
            base_list.append(f'{b}')

        catid = url.split('/')[-1]

        dataurl_list = ['file', 'get', f'{catid}']

        d_elements = base_list + dataurl_list
        dataurl = '/'.join(d_elements)
        download_and_save_file(dataurl, extension='.zip', output_dir=gis_dir, gis_id='USGS019')

    if url.startswith('https://mrdata.usgs.gov/'):
        get_available_files(url, gis_dir=os.path.join('../..', 'USGS_osd_gravity'), gis_id='USGS_osd_gravity')

    # url = 'https://www.sciencebase.gov/catalog/file/get/63a20e8cd34e176674f51d51'
    # get_available_files(url)

    # url = 'https://gdr.openei.org/files/954/Tularosa_170511_mdl13_cellcenter.dat'
    # download_and_save_file(url, extension='.zip', output_dir='out')


if __name__ == '__main__':
    main()
