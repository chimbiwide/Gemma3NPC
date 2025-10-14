import csv
import json

def create_csv(csv_name:str):
    data = ["Extracted Info", "Location Description", "Character Bio", 'System Prompt']
    with open(csv_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        print("created csv file")

def write_csv(csv_name: str,  data: list):
    with open(csv_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i, row in enumerate(data):
            writer.writerow(row)
            print(f"written row {i}")


def read_files(info: str, location_description: str, char_bio: str, system_prompt:str) -> list:
    data = []
    with open (info, 'r', encoding='utf-8') as f1, \
        open(location_description, 'r', encoding='utf-8') as f2, \
        open(char_bio, 'r', encoding='utf-8') as f3, \
        open(system_prompt, 'r', encoding='utf-8') as f4:

        for l1, l2, l3, l4 in zip(f1,f2,f3,f4):
            info_data = l1.strip()
            location_data = l2.strip()
            bio_data = l3.strip()
            prompt_data = l4.strip()
            row = [info_data, location_data,bio_data,prompt_data]
            data.append(row)

    return data

def main():
    filename = "Generated_info.csv"
    create_csv(filename)
    data = read_files("info.txt", "location_description.txt", "char_bio.txt", "system_prompt.txt")
    write_csv(filename,data)
    print("DONE")

if __name__ == "__main__":
    main()