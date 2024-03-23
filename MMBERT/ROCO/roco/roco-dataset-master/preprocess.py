import os
from PIL import Image
# val_err={'ROCO_01878', 'ROCO_03373', 'ROCO_02228', 'ROCO_03483', 'ROCO_00187', 'ROCO_00495', 'ROCO_00890', 'ROCO_01167', 'ROCO_00299', 'ROCO_02557', 'ROCO_01132', 'ROCO_01553', 'ROCO_03392', 'ROCO_00191', 'ROCO_02962', 'ROCO_02408', 'ROCO_00656', 'ROCO_02293', 'ROCO_00113', 'ROCO_00838', 'ROCO_03926', 'ROCO_00186', 'ROCO_00707', 'ROCO_03570', 'ROCO_02757', 'ROCO_00593', 'ROCO_01675', 'ROCO_00415', 'ROCO_00108', 'ROCO_03803', 'ROCO_01852', 'ROCO_00227', 'ROCO_02175'}
# test_err={'ROCO_01437', 'ROCO_02769', 'ROCO_00190', 'ROCO_00319', 'ROCO_02081', 'ROCO_03270', 'ROCO_01908', 'ROCO_01062', 'ROCO_03672', 'ROCO_01082', 'ROCO_03504', 'ROCO_04095', 'ROCO_04807', 'ROCO_01629', 'ROCO_00515', 'ROCO_01409', 'ROCO_00599', 'ROCO_01544', 'ROCO_04207', 'ROCO_01761', 'ROCO_01737', 'ROCO_01425', 'ROCO_04689', 'ROCO_01651', 'ROCO_04283', 'ROCO_01873', 'ROCO_00036', 'ROCO_04425', 'ROCO_03804', 'ROCO_04184', 'ROCO_01326', 'ROCO_00587', 'ROCO_02537'}
#train_err={'ROCO_00216', 'ROCO_00103', 'ROCO_00007', 'ROCO_00080', 'ROCO_00440', 'ROCO_00011', 'ROCO_00346', 'ROCO_00146', 'ROCO_00615', 'ROCO_00245', 'ROCO_00467', 'ROCO_00121', 'ROCO_00371', 'ROCO_00548', 'ROCO_00545', 'ROCO_00259', 'ROCO_00541', 'ROCO_00318', 'ROCO_00230', 'ROCO_00320', 'ROCO_00383', 'ROCO_00439', 'ROCO_00542', 'ROCO_00524', 'ROCO_00238', 'ROCO_00239', 'ROCO_00102', 'ROCO_00352', 'ROCO_00365', 'ROCO_00292', 'ROCO_00393', 'ROCO_00359', 'ROCO_00657'}
with open("data/validation/validation.txt","w") as file:
    pass
f = open("data/validation/radiology/captions.txt")
lines = f.readlines()#读取全部内容
for line in lines:
    line=line.strip('\n')
    two=line.split(' ',1)
    two[0]=two[0].strip('\t')
    img="data/validation/radiology/images/"+two[0]+'.jpg'
    caption=two[1]
    if os.path.exists(img):
        file = open('data/validation/validation.txt', mode='a+')
        file.write(two[0] + ' ' + caption + '\n')
        file.close()
        
        # try:
        #     img = Image.open(img).convert('RGB')
        #     img.verify()
        #     img.close()
        # except:
        #     print('损坏: %s' % two[0])
        # else:
        #     file = open('data/test/test.txt', mode='a+')
        #     file.write(two[0] + ' ' + caption + '\n')
        #     file.close()



