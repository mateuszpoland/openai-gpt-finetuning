import json

class JSONLTrainingDataProcessor:
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
        Assistant is a large language model trained by OpenAI. Assistant is designed to act as a named entity recognition engine for Romanian addresses. It analyzes input strings that may contain spelling mistakes, be incomplete, or improperly formatted. Using its knowledge of Romanian administrative structures, it extracts and classifies components of the address. The components it is expected to extract are: street, house number, flat number, block number, staircase number, apartment number, postcode, county, commune, village name or city name
        """

        user_message = f"""
        TOOLS 
-------
Assistant can look to attached file, which is it's knowledge base to look for matching postcode that it extracts from the user input. If postcode is present in the user input and can be found directly in one of the json objects in knowledge base, then the Assistant will be able to extract with 100% confidence the actual county that address is placed in, as well as the place name, which can be either city or village situated in a particular commune. 
Assistant will then produce the output based on data extracted from knowledge base.
If there is no postcode in the address, then Assistant will not take any information from knowledge base and rely solely upon it's expert knowledge about Romanian administrative structure.


RESPONSE FORMAT INSTRUCTIONS
--------------------
When responding to me, please output a response in ONLY one format, which is json string containing extracted parts of the Romanian address provided as a user input.

```json
{{
	"street": string  // The street name extracted from Romanian address.
	"house": string  // The house number extracted from Romanian address.
	"flat": string  // The flat number extracted from Romanian address.
	"block": string  // The block number extracted from Romanian address.
	"staircase": string  // The staircase number/mark extracted from Romanian address.
	"staircase": string  // The floor number/mark extracted from Romanian address.
	"apartment": string  // The apartment number extracted from Romanian address.
	"landmark": string  // Additional data contained in the address string, but not part of standard address component. May be valuable for delivery or navigation purposes to identify exact location
	"intercom": string  // The intercom number extracted from Romanian address.
	"postcode": string  // The postcode extracted from Romanian address. 
	"county": string  // The county extracted or inferred from Romanian address.
	"commune": string  // The commune name extracted from the Romanian address.
	"village": string  // The village name extracted from the Romanian address. 
	"city": string  // The city extracted or inferred from Romanian address.
}}
```

--------------------
USER'S INPUT
--------------------
Here is the user's input (remember to respond with a structured json string with a structure shown above, and NOTHING else):
{input_address}
"""

        return {
            "messages": [
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
                {"role": "assistant", "content": assistant_response}
            ]
        }

    def clean_empty_fields(data):
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
                    clean_empty_fields(value) # Recursively clean nested dictionaries
        
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