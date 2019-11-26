from SPARQLWrapper import SPARQLWrapper, JSON

sparqlDBpedia = SPARQLWrapper("http://dbpedia.org/sparql")
 
def getAnimals(animalType):
    sparqlDBpedia.setQuery("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?animal ?thumbnail
        WHERE { ?animal <http://purl.org/linguistics/gold/hypernym> dbr:"""+animalType+""".
        ?animal dbo:thumbnail ?thumbnail.}
    """)
    sparqlDBpedia.setReturnFormat(JSON)
    results = sparqlDBpedia.query().convert()
    results = results["results"]["bindings"]
    animalList = []
    for result in results:
        name = result["animal"]["value"].split("/")[-1].replace("_"," ")
        animal = {"name":name, "uri":result["animal"]["value"], "thumbnail":result["thumbnail"]["value"]}
        animalList.append(animal)
    return animalList

if __name__ == '__main__':
    resultJson = getAnimals("Dog")
    print(resultJson)