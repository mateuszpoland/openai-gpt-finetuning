from dataclasses import dataclass
from .prompt_v4 import Prompt

class PromptV1(Prompt):
    """Romanian Address NER Prompt Version 1"""
    
    def __init__(self):
        system_message = """Given in input a string representing Romanian address typed in randomly by user, containing spelling mistakes, wrong formatting and missing data, \
extract and tag each part of the input following instruction below.
Remember, for each entry, use the expert knowledge about Romanian administrative structure.Requirements:
1. For each address, add-in county.
2. For each address, consider if it is address in commune. If that is the case, fill in commune and village entries. For the city, add the nearby city that the commune and village belongs to.
If the input represents an address in the city itself, skip the commune and village parts and fill in only city."""

        user_message_template = """<< FORMATTING >>
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{{
    "street": string  // The street name extracted from Romanian address. Examples: Averescu Alexandru Maresal, Penes Curcanul, I.C. Bratianu, 9 Mai
    "house": string  // The house number extracted from Romanian address. Examples: 21, 78A, 155, 4
    "flat": string  // The flat number extracted from Romanian address. Examples: 85C/1 then flat is 1, 75/115 then flat is 115 etc.
    "block": string  // The block number extracted from Romanian address. Examples: bl. 2 then block is 2, BL V2, then block is V2, bl.2C, then block is 2C, BL 441, then block is 441 etc.
    "staircase": string  // The staircase number/mark extracted from Romanian address. Examples: sc. 3 then staircase is 3, sc.A, then staircase is A, SC B, then staircase is B etc.
    "staircase": string  // The floor number/mark extracted from Romanian address. Examples: et. 3 then floor is 3, et.5, then floor is A etc.
    "apartment": string  // The apartment number extracted from Romanian address. Examples: ap. 489 then apartment is 489, ap. 79 8-14, then apartment is 79 8-14 etc.
    "landmark": string  // Additional data contained in the address string, but not part of standard address component. May be valuable for delivery or navigation purposes to identify exact location. Examples: [EASYBOX PIATA VASILE AARON] STR. SEMAFORULUI, 7, SIBIU, then landmark is Easybox Piata Vasile Aaron
    "intercom": string  // The intercom number extracted from Romanian address. Examples: INTERFON DAMIAN 5, then intercom is 5 etc.
    "postcode": string  // The postcode extracted from Romanian address. Examples: 107402, 800620, 41832, 117710.
    "county": string  // The county extracted or inferred from Romanian address. Examples: Bacău, Oradea, Brașov, Buzău etc.
    "commune": string  // The commune name extracted from the Romanian address. Examples: COM BUCOV, then commune is Bucov etc.
    "village": string  // The village name extracted from the Romanian address. Examples: SAT CHITORANI, then village is Chitorani
    "city": string  // The city extracted or inferred from Romanian address. Examples: Iași, Bucuresti, Craiova, Oradea etc.
}}
```
<< INPUT >>
{input_address}
<< OUTPUT (remember to include the ```json)>>


Note: Do not rely on the examples as templates for your answers. Instead, use them to understand the format of the output. Each input should be processed based on its own structure and content.

IMPORTANT! Carefully analyze each address and generate the output based on its unique structure, even if it slightly differs from the examples.

Remember, each address input is different and should be analyzed on its own. Do not simply replicate the format or content of the examples.
IMPORTANT!!!
In the attached knowledge base, there are json objects representing locations bound to particular postcode. Upon lookup in external database, there are several possible locations of this particular address along with information about county, place name and postal code:
Remember!!!
 If 100% confident, EXTRACT THE postal_code and COUNTY from the json object and save them as postcode and county in the output. If not 100% confident, skip extracting information from provided additional data.
!!!
Remember, change all UPPERCASE into first Uppercase letter and following lowercase, like: EXAMPLE STREET NAME becomes: Example Street Name
Remember, for each entry, use the expert knowledge about Romanian administrative structure. For each address, add-in county. For each address, consider if it is address in commune. If that is the case, fill in commune and village entries. For the city, add the nearby city that the commune and village belongs to.
Remember, if the input has spelling mistakes and/or typos, you need to fix them so that they represent correct entities (cities, streets, villages etc.) in Romania. If you can't reliably determine the true name based on the name with typo, return string ERROR in place of the entity instead.
If the input represents an address in the city itself, skip the commune and village parts and fill in only city."""
        
        super().__init__(
            version="v1",
            system_message=system_message,
            user_message_template=user_message_template
        )