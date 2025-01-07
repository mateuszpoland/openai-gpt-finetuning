import json

class JSONLAddressKnowledgeBaseProcessor:
    def __init__(self, input_file_path: str, output_file_path: str) -> None:
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
    
    def process(self):
        with open(self.input_file_path, 'r', encoding='utf-8') as input_file, open(self.output_file_path, 'w', encoding='utf-8') as output_file:
            data = json.load(input_file)  # Assuming the whole file is one large JSON object
            ro_places = data.get("ro_places", [])
            
            for place in ro_places:
                # Construct the summary string
                summary = self._generate_summary(place)
                # Write the summary to the output file with a new line
                output_file.write(summary + "\n\n")
            
    def _generate_summary(self, place):
        postal_code = place.get("postal_code", "")
        place_name = place.get("place_name", "")
        county = place.get("county", "")
        summary = f"{postal_code} is postalcode for the place name {place_name} and the address is situated in {county} county.\n\n"

        return summary

if __name__ == "__main__":
    processor = JSONLAddressKnowledgeBaseProcessor('RO_names_postalcodes.json', 'RO_names_postalcodes.txt')
    processor.process()