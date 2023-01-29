import configparser
import csv
import os


def create():
    path = os.path.dirname(os.path.abspath(__file__))    
    with open('manifest.csv', 'w', newline='') as file:
        
        # write the header for the manifest
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['name', 'description', 'location'])
        
        # walk through current folder to find valid domains
        for dirpath, _, filenames in os.walk(path):
            if 'domain.info' in filenames:
                infopath = os.path.join(dirpath, 'domain.info')
                config = configparser.RawConfigParser()
                config.optionxform = str 
                config.read(infopath)
                general = dict(config.items('General'))
                name = general.get('name', None)
                desc = general.get('description', None)
                loc = dirpath[len(path):]
                loc = loc.replace('\\', '/') + '/'
                writer.writerow([name, desc, loc])


def load():
    EXP_DICT = {}
    path = os.path.dirname(os.path.abspath(__file__))
    path_to_manifest = os.path.join(path, 'manifest.csv')
    with open(path_to_manifest) as file:
        reader = csv.reader(file, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                row = [cell.replace('\'', '') for cell in row]
                key, *entries = row
                EXP_DICT[key] = tuple(entries) + (None,)
    return EXP_DICT


if __name__ == '__main__':
    create()
