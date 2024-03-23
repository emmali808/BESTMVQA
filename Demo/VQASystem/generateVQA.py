from ner import ner

data="History of present illness: due to accidental fall at home 5 hours ago, injury to the left upper arm, immediate left upper arm pain, swelling, functional limitation and obvious angulation, limited dorsal extension of the left thumb, and numbness of the left finger. There was no coma, nausea and vomiting at the time of the injury. After the injury, his family members were rushed to the emergency department of our hospital. After careful examination and medical history in the outpatient department, X-ray examination was planned to be diagnosed as left humerus fracture and radial nerve injury and admitted to the department. initial diagnosis:1. Comminuted fracture of the left humerus.2. Left radial nerve injury.3. Sequela of cerebral infarction.Epidural Marcaine for postoperative analgesia."

entities = ner(text=data)
for entity in entities:
    if entity[1]=='DISEASE':
        print("DISEASE:")
        print(entity[0])
        # entDic['DISEASE'].append(entity[0])
    elif entity[1]=='CHEMICAL':
        print("CHEM:")
        print(entity[0])
        # entDic['CHEMICAL'].append(entity[0])