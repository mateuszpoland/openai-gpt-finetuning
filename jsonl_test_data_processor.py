import json

class JSONLTestDataProcessor:
    def __init__(self, input_file_path: str, output_file_path: str) -> None:
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def process(self):
        with open(self.input_file_path, 'r', encoding='utf-8') as input_file, open(self.output_file_path, 'w') as output_file:
            while True:
                input_line = input_file.readline()
                if not input_file:
                    break #end of file
                output_line = input_file.readline()
                if not output_line:
                    break #end of file
                
                result = self.process_input_output_pair(input_line, output_line)
                print(result)
                result = self.format_message(result)
                json.dump(result, output_file)
                output_file.write('\n')
    
    def process_input_output_pair(self, input_line: str, output_line: str):
        input_address = input_line.strip().replace('input:', '').strip()
        assistant_response = output_line.strip().replace('output:', '').strip()

        system_message = """
        Your task is to act as a named entity recognition engine for Romanian addresses. Extract and classify components of the provided address string, such as street, house number, flat number, block number, staircase number, apartment number, postcode, county, commune, village name, city name.
        If the address contains a postcode, check against the attached knowledge base.
        Upon finding a postcode match, extract the county and place name (city or village) with full confidence.
        Output only in the specified JSON format.

        # Output Format

        Provide the extracted address details as a JSON string:
        ```
        {
            "street": string,  
            "house": string,  
            "flat": string,  
            "block": string,  
            "staircase": string,  
            "floor": string,  
            "apartment": string,  
            "landmark": string,  
            "intercom": string,  
            "postcode": string,  
            "county": string,  
            "place name": string
        }
        ```
        """

        user_message = f"""
        {input_address}
        """

        return {
            "messages": [
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def clean_empty_fields(self, data):
        """Recursively clean empty string values in the dictionary."""
        if isinstance(data, dict):
            for key, value in data.items():
                if value == "":
                    data[key] = ''  # Replace empty string with single quotes (as a plain string)
                elif value == '"\"':
                    data[key] = ''
                elif value == '{{':
                    data[key] = '{'
                elif isinstance(value, dict):
                    self.clean_empty_fields(value) # Recursively clean nested dictionaries
        
        return data

    def format_message(self, entry):
        """Reformat a single message entry to ensure consistent JSON formatting."""
        messages = entry['messages']
        for message in messages:
            if message['role'] == 'user':
                message['content'] = message['content'].replace('it\'s', 'its')  # Correct common grammar mistake
                json_snippet_index = message['content'].find('```json')
                if json_snippet_index != -1:
                    json_start = message['content'].find('{', json_snippet_index)
                    json_end = message['content'].rfind('}') + 1
                    json_content = message['content'][json_start:json_end]
                    try:
                        json_object = json.loads(json_content)
                        formatted_json = json.dumps(json_object, indent=4)
                        message['content'] = message['content'][:json_start] + formatted_json + message['content'][json_end:]
                    except json.JSONDecodeError:
                        print("Failed to decode JSON content in user message.")
                elif message['role'] == 'assistant':
                    try:
                        content = json.loads(message['content'])
                        corrected = content.replace('"', "'")
                        cleaned = self.clean_empty_fields(corrected)
                        json_object = json.loads(cleaned)
                        message['content'] = json.dumps(json_object, indent=4)
                    except json.JSONDecodeError:
                        print("Failed to decode JSON content in assistant message.")
        
        return entry
