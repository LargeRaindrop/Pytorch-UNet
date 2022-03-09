import json
import os
from glob import glob

from tqdm import tqdm


def main():
    json_list = glob(os.path.join('own_data', 'raw', '*.json'))
    for json_file in tqdm(json_list):
        # print(json_file)
        with open(json_file) as fd:
            content = json.load(fd)
        for item in content['shapes']:
            if item['label'] == 'zigong':
                content['shapes'].remove(item)
            elif item['shape_type'] != 'polygon':
                item['shape_type'] = 'polygon'
        with open(json_file, 'w') as fd:
            json.dump(content, fd, indent=4)


if __name__ == '__main__':
    main()
