import os.path

test_files = ['1632383745473', '1635866721784', '1637757885630', '1649402935713', '1632651622683', 
'1649062565572', '1633322279185', '1635864966113', '1649935754742', '1630759437618', '1630764595166', 
'1631815347745', '1631812405081', '1649437939395', '1643882281893', '1643731875413', '1633327203891', 
'1644860873977', '1633263809291', '1630761956607', '1636114931710', '1635867004228', '1634559711780', 
'1648807289449', '1649762777966', '1630958295720', '1636130267317', '1633353472431', '1645482982124', 
'1635866676539', '1630763519305', '1631811368362', '1636471755761', '1636561211555', '1648732170485', 
'1648548189287', '1629916830485', '1643751348983', '1632083385402', '1638802723937', '1635516181904', 
'1636471694209', '1648554684882', '1648807488100', '1631825039027', '1636560905891', '1643732613591', 
'1631022389311', '1642977594311', '1638085804782', '1633321409878', '1635867106899', '1644239828342', 
'1643064753832', '1635866958281', '1644857432421', '1637757496275', '1641506827471', '1632992399268', 
'1633262946604', '1638183148777', '1636109572562', '1637757834495', '1637757785867', '1636107781115', 
'1641557443523', '1644856919350', '1644861166020', '1635866642974', '1633552603095', '1632137764275', 
'1632335440007', '1635867057506', '1630674474162'
]

for file_name in test_files:

    data1 = data2 = data3 = data4 = data5 = ""

    if os.path.isfile("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript1.txt"):
        fp = open("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript1.txt")
        data1 = fp.read()

    if os.path.isfile("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript2.txt"):
        fp = open("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript2.txt")
        data2 = fp.read()
    
    if os.path.isfile("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript3.txt"):
        fp = open('/home/pooja/pooja/unil/transcripts/transcripts/'+file_name+'Transcript3.txt')
        data3 = fp.read()
    
    if os.path.isfile("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript4.txt"):
        fp = open('/home/pooja/pooja/unil/transcripts/transcripts/'+file_name+'Transcript4.txt')
        data4 = fp.read()
    
    if os.path.isfile("/home/pooja/pooja/unil/transcripts/transcripts/"+file_name+"Transcript5.txt"):
        fp = open('/home/pooja/pooja/unil/transcripts/transcripts/'+file_name+'Transcript5.txt')
        data5 = fp.read()

    data1 += " "
    data1 += data2
    data1 += " "
    data1 += data3
    data1 += " "
    data1 += data4
    data1 += " "
    data1 += data5

    with open ('/home/pooja/pooja/unil/transcripts/merged_test_transcripts/'+file_name+'merged.txt', 'w') as fp:
        fp.write(data1)
