import json

def ground_truth_stats (data):
    wrong_count = 0
    right_count = 0
    contradictory_count = 0
    partially_right_count = 0

    for i in data:
        for j in i["messages"]:
            if j["criteria_ground_truth"] == 0:
                wrong_count += 1
            if j["criteria_ground_truth"] == 1:
                right_count += 1
            if j["criteria_ground_truth"] == 2:
                contradictory_count += 1
            if j["criteria_ground_truth"] == 3:
                partially_right_count += 1
    
    print("Wrong annotations:\t\t\t" + str(wrong_count))
    print("Right annotations:\t\t\t" + str(right_count))
    print("Contradictory annotations:\t\t" + str(contradictory_count))
    print("Partially right annotations:\t\t" + str(partially_right_count))
    print("______________________________________________________")

def coreference_stats (data):
    wrong_count = 0
    right_count = 0
    partially_right_count = 0

    for i in data:
        if i["coreference_resolution_criteria"] == 0:
            wrong_count += 1
        if i["coreference_resolution_criteria"] == 1:
            right_count += 1
        if i["coreference_resolution_criteria"] == 2:
            partially_right_count += 1
        
    print("Wrong coreference chains:\t\t" + str(wrong_count))
    print("Right coreference chains:\t\t" + str(right_count))
    print("Partially right coreference chains:\t" + str(partially_right_count))
    print("______________________________________________________")

def openie_stats (data):
    wrong_count = 0
    right_count = 0
    partially_right_count = 0

    for i in data:
        for j in i["messages"]:
            if j["criteria_openIE"] == 0:
                wrong_count += 1
            if j["criteria_openIE"] == 1:
                right_count += 1
            if j["criteria_openIE"] == 2:
                partially_right_count += 1
            if j["criteria_openIE"] == 3:
                partially_right_count += 1
    
    print("Wrong triples:\t\t\t\t" + str(wrong_count))
    print("Right triples:\t\t\t\t" + str(right_count))
    print("Partially right triples:\t\t" + str(partially_right_count))
    print("______________________________________________________")

def spn4re_stats (data):
    wrong_count = 0
    right_count = 0
    partially_right_count = 0

    for i in data:
        for j in i["messages"]:
            if j["criteria_SPN4RE"] == 0:
                wrong_count += 1
            if j["criteria_SPN4RE"] == 1:
                right_count += 1
            if j["criteria_SPN4RE"] == 2:
                partially_right_count += 1
    
    print("Wrong triples:\t\t\t\t" + str(wrong_count))
    print("Right triples:\t\t\t\t" + str(right_count))
    print("Partially right triples:\t\t" + str(partially_right_count))
    print("______________________________________________________")

def entity_recognition_stats (data):
    wrong_count = 0
    right_count = 0
    partially_right_count = 0

    for i in data:
        for j in i["messages"]:
            if j["linked_entity_criteria"][0] == 0:
                wrong_count += 1
            if j["linked_entity_criteria"][0] == 1:
                right_count += 1
            if j["linked_entity_criteria"][0] == 2:
                partially_right_count += 1
    
    print("Wrong entities recognised:\t\t" + str(wrong_count))
    print("Right entities recognised:\t\t" + str(right_count))
    print("Partially right entities recognised:\t" + str(partially_right_count))
    print("______________________________________________________")

def entity_linking_stats (data):
    wrong_count = 0
    right_count = 0
    partially_right_count = 0

    for i in data:
        for j in i["messages"]:
            if j["linked_entity_criteria"][1] == 0:
                wrong_count += 1
            if j["linked_entity_criteria"][1] == 1:
                right_count += 1
            if j["linked_entity_criteria"][1] == 2:
                partially_right_count += 1
    
    print("Wrong entities linked:\t\t\t" + str(wrong_count))
    print("Right entities linked:\t\t\t" + str(right_count))
    print("Partially right entities linked:\t" + str(partially_right_count))
    print("______________________________________________________")

if __name__ == '__main__':
    f = open('sample_annotated.json')
    data = json.load(f)
    f.close()

    print("______________________________________________________")
    print("GROUND TRUTH TRIPLE ANNOTATIONS")
    ground_truth_stats(data)

    print("COREFERENCE RESOLUTION")
    coreference_stats(data)

    print("OPENIE TRIPLES")
    openie_stats(data)

    print("SPN4RE TRIPLES")
    spn4re_stats(data)

    print("ENTITY RECOGNITION")
    entity_recognition_stats(data)

    print("ENTITY LINKING")
    entity_linking_stats(data)