import xml.etree.ElementTree as ET
import csv
import os
from pathlib import Path
import re

def natural_sort_key(filename):                                                                 #sort key for natural sorting of filenames (ensures ADNI_002_S_0413 comes before ADNI_023_S_0376)
    parts = re.split(r'(\d+)', str(filename))                                                   #extract all numbers and text separately
    return [int(part) if part.isdigit() else part.lower() for part in parts]                    #convert numeric parts to integers for proper numerical sorting

def extract_xml_data(xml_file):                                                                 #extract all relevant data from an XML file and return as a dictionary.
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = {}
    data['filename'] = os.path.basename(xml_file)                                               #add filename
    
    for element in root.iter():                                                                    #remove namespace prefix for easier parsing
        if '}' in element.tag:
            element.tag = element.tag.split('}', 1)[1]
    
    project = root.find('.//project')                                                           #project level
    if project is not None:
        data['projectIdentifier'] = project.findtext('projectIdentifier', default='')
        data['projectDescription'] = project.findtext('projectDescription', default='')
        data['siteKey'] = project.findtext('siteKey', default='')
    
    subject = root.find('.//subject')                                                           #subject level
    if subject is not None:
        data['subjectIdentifier'] = subject.findtext('subjectIdentifier', default='')
        data['researchGroup'] = subject.findtext('researchGroup', default='')
        data['subjectSex'] = subject.findtext('subjectSex', default='')
        
        for info in subject.findall('subjectInfo'):                                             #subject info items
            item_name = info.get('item', '')
            if item_name:
                data[item_name] = info.text or ''
    
    visit = root.find('.//visit')                                                               #visit level
    if visit is not None:
        data['visitIdentifier'] = visit.findtext('visitIdentifier', default='')
        
        for assessment in visit.findall('assessment'):                                          #assessments
            assessment_name = assessment.get('name', '')
            for component in assessment.findall('component'):
                component_name = component.get('name', '')
                for score in component.findall('assessmentScore'):
                    attribute = score.get('attributeibute', '')
                    if attribute:
                        data[attribute] = score.text or ''
    
    study = root.find('.//study')                                                               #study level
    if study is not None:
        data['studyIdentifier'] = study.findtext('studyIdentifier', default='')
        data['subjectAge'] = study.findtext('subjectAge', default='')
        data['ageQualifier'] = study.findtext('ageQualifier', default='')
        data['weightKg'] = study.findtext('weightKg', default='')
        data['postMortem'] = study.findtext('postMortem', default='')
    
    series = root.find('.//series')
    if series is not None:
        data['seriesIdentifier'] = series.findtext('seriesIdentifier', default='')              #series level
        data['modality'] = series.findtext('modality', default='')
        data['dateAcquired'] = series.findtext('dateAcquired', default='')
    
    derived = root.find('.//derivedProduct')                                                    #derived product
    if derived is not None:
        data['imageUID'] = derived.findtext('imageUID', default='')
        data['processedDataLabel'] = derived.findtext('processedDataLabel', default='')
        data['imageType'] = derived.findtext('imageType', default='')
        data['tissue'] = derived.findtext('tissue', default='')
        data['hemisphere'] = derived.findtext('hemisphere', default='')
        data['anatomicStructure'] = derived.findtext('anatomicStructure', default='')
        data['registration'] = derived.findtext('registration', default='')
        data['creationDate'] = derived.findtext('creationDate', default='')
        
        rating = derived.find('imageRating')                                                    #image rating
        if rating is not None:
            data['imageRating_description'] = rating.findtext('ratingDescription', default='')
            data['imageRating_value'] = rating.findtext('value', default='')
    
    originalImage = root.find('.//originalRelatedImage')                                           #original related image
    if originalImage is not None:
        data['originalImageUID'] = originalImage.findtext('imageUID', default='')
        data['originalImageDescription'] = originalImage.findtext('description', default='')
        
        for protocol in originalImage.findall('.//protocol'):                                      #protocol terms
            term = protocol.get('term', '')
            if term:
                data[f'protocol_{term}'] = protocol.text or ''
    return data

def process_xml_directory(directory_path, output_csv='output.csv'):                             #process all XML files in a directory and write to CSV by takingthe path to directory containing XML files and the name of output CSV fil
    xml_files = list(Path(directory_path).glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found")
        return
    
    xml_files.sort(key=natural_sort_key)                                                        #sort XML files naturally by filename
    
    all_data = []                                                                               #extract data from all XML files
    for xml_file in xml_files:
        print(f"Processing: {xml_file.name}")
        try:
            data = extract_xml_data(xml_file)
            all_data.append(data)
        except Exception as e:
            print(f"Error in {xml_file.name}: {e}")
    
    if not all_data:
        print("No data")
        return
    
    all_columns = set()                                                                         #get all unique column names
    for data in all_data:
        all_columns.update(data.keys())
    
    columns = ['filename'] + sorted([col for col in all_columns if col != 'filename'])          #sort columns for consistent output (filename first, then alphabetically)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:                        #write to CSV
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Total columns: {len(columns)}")

if __name__ == "__main__":
    directory = "."                                                                             #specify the directory containing XML files
    process_xml_directory(directory, output_csv='combined_data.csv')                            #process all XML files and create CSV