from igdb.wrapper import IGDBWrapper
from jsonschema import validate
import json

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer"
            },
            "name": {
                "type": "string"
            },
            "summary": {
                "type": "string"
            },
            "genres": {
                "id": {
                    "type": "integer"
                },
                "name": {
                    "type": "string"
                }
            }
        }
    }
}


wrapper = IGDBWrapper("op1z4ho7doyt2morgsh11wezj9yjc4", "eddsix0pyczrhnghvhidqworxgv4od")
byte_array = wrapper.api_request(
    "games",
    "fields name, summary, genres.name; sort follows desc; limit 500; where version_parent = null & summary != null & platforms = (48, 49, 6) & genres != null;"
)
data_str = byte_array.decode("utf-8")
data = json.loads(data_str)
validate(instance=data, schema=schema)



filename = "games.json"
with open(filename, "w") as f:
    json.dump(data, f, indent=2)

with open(filename, "r") as f:
    data = json.load(f)
    final_json = []
    for game in data:
        final_json.append({
            "id": game["id"],
            "name": game["name"],
            "summary": game["summary"],
            "tags": [genre["name"] for genre in game["genres"]]
        })
    with open("games_clean_mini.json", "w") as f:
        json.dump(final_json, f, indent=2)
        