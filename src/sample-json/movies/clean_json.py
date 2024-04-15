import json
import ast

filename = "movies.json"

with open(filename, "r") as f:
    data = json.load(f)
    final_json = []
    summary_lengths = []
    for movie in data:

        keywords_list = ast.literal_eval(movie["keywords"])
        summary = set()
        summary_length = 0
        for keyword in keywords_list:
            words = keyword['name'].split(" ")
            for word in words:
                if summary_length < 50:
                    summary_length += 1
                    summary.add(word)
        summary_lengths.append(len(summary))

        final_json.append({
            "id": int(movie["id"]),
            "name": movie["title"],
            "summary": " ".join(list(summary)),
            "tags": [genre["name"] for genre in ast.literal_eval(movie["genres"])]
        })

    with open("movies_clean.json", "w") as f:
        json.dump(final_json, f, indent=2)

    summary_lengths.sort()
    print("Summary lengths: ", summary_lengths[490:])