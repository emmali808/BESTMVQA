total_strs=[
    r".*change (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*change (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*modify (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*revise (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*amend (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*emend (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*alter (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*",
    r".*change (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
    r".*modify (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
    r".*revise (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
    r".*amend (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
    r".*emend (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
    r".*alter (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b) to (?:[t][h][e] )?(\b(?:VQA-RAD|SLAKE|MED-VQA|PATHVQA)\b).*",
]
# total_strs=[
#     r".*change (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\S+).*",
#     r".*change (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
#     r".*modify (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
#     r".*revise (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
#     r".*amend (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
#     r".*emend (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
#     r".*alter (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b[\w]+).*",
# ]
half_strs=[
    r".*change (?:[t][h][e] )?(\b[\w]+).*",
    r".*modify (?:[t][h][e] )?(\b[\w]+).*",
    r".*revise (?:[t][h][e] )?(\b[\w]+).*",
    r".*amend (?:[t][h][e] )?(\b[\w]+).*",
    r".*emend (?:[t][h][e] )?(\b[\w]+).*",
    r".*alter (?:[t][h][e] )?(\b[\w]+).*",
]
test_str=[
    r".*change (?:[t][h][e] )?(.*?) to (?:[t][h][e] )?(\b(?:MEVF|CR|MMBERT|VQAMIX|MMQ|CSMA|PTUNIFIER|TCL|METER|MEDVINT|MINIGPT-4)\b).*"
]