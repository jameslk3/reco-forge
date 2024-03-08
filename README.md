# reco-forge project
## Group name: Trailblazing bananas  
### Group members: Juno Kim (jkim826), James Kendrick (jameslk3)

Introduction:

    - Decription: reco-forge is a crate which provides an interface for users to generate personalized recommendation systems.

    - Goals/objectives: Learn about how Natural Language processing works, how that can be used to improve recommendation systems, and using serialization/deserialization to construct nodes for our algorithm.

    - Why we chose this idea: We chose this idea because we wanted to learn about Natural Language Processing and interesting data structures and algorithms associated with recommendation systems. Also this idea is especially applicable to topics like video games and movies which we are both interested in but it can be applied to any group of things that each have descriptions.

Technical Overview:

    Completed by Checkpoint 1 (4/3 - 4/7):
        - To initialize the data structure that we will use as the basis for the recommendation system, we require that the user provides their dataset in a specified JSON schema. This schema will contain the name of the thing, it's description, and optionally, a list of tags that can be used for filtering.
        - We will use deserialize the JSON into nodes which will each represent one item.
        - Nodes will have a field for a vector representation of the description. These embeddings will be generated using a pre-trained natural language processing model.
    
    Completed by Checkpoint 2 (4/17 - 4/21) or Final Due Date (5/1):
        - We will then construct the data structure that will be operated on by the algorithm of our choice, this we are still deciding on. The current idea is a graph based algorithm containing clusters. The formation of the clusters would be determined by similarities between nodes using a measure like cosine similarity or euclidean distance.
        - After the recommendatiuon system has been created, we will have functions part of our crate that allow the user to interact with the system. The input to get a recommendation back will be a description. This description will be transformed into a dense vector and then our algorithm will return similar nodes. Optionally, and tag/s contained in any of the nodes can be used to filter results. If time permits, another type of input can be allowing the user to select things from the dataset they like and then our system can skip straight to finding those nodes and determining similar things.
    
Possible Challenges:

    - Choosing the best data structure and algorithm that allows for accurate recommendations across a wide-range of categories, high performace, and reasonable complexity.
    - Figuring out how to interact with pre-trained NLP models and tools in Rust.