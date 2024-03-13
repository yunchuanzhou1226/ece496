'''
RUN the following command in virtuoso 
loadi("/groups/czzzgrp/anaconda3/envs/my-env/lib/python3.11/site-packages/skillbridge/server/python_server.il")
pyStartServer
'''


from skillbridge import Workspace


def edit_schematic(inst_name, properties, 
                   lib_name = "Capstone_Project", 
                   lib_cell_name = "Candence_Python_Bridge_Testing", 
                   lib_cell_view_name = "schematic"):
    # open the schematic
    ws = Workspace.open()
    cell_view = ws.db.open_cell_view_by_type(lib_name, lib_cell_name, lib_cell_view_name, "", "a")
    insts = cell_view.instances

    # change the instance of the given name
    for inst in insts:
        if inst.name == inst_name:
            info = "INFO: In lib: {} cell: {} view: {} modified {} ".format(lib_name, lib_cell_name, lib_cell_view_name, inst_name)
            
            # change every property listed in the dictionary
            for prop in properties:
                if properties[prop] != '' and inst[prop] != None:
                    inst[prop] = properties[prop]
                    info += "{}: {} ".format(prop, inst[prop])
            print(info)
            break
    
    # check and save the schematic
    ws.db.check(cell_view)
    ws.db.save(cell_view)

def get_net_edge_index(node_mapping,
                       lib_name = "Capstone_Project", 
                       lib_cell_name = "Candence_Python_Bridge_Testing", 
                       lib_cell_view_name = "schematic"):
    ws = Workspace.open()
    cell_view = ws.db.open_cell_view_by_type(lib_name, lib_cell_name, lib_cell_view_name, "", "a")
    nets = cell_view.nets
    # print(dir(cell_view),'\n')

    edge_index = []
    for net in nets:
        info = "Net: {} is connected to instances: ".format(net.name)
        instances = net.instTerms
        for inst in instances:
            info += "{} ".format(inst.inst.name)
        print(info)

        for i in range(len(instances)):
            for j in range(i+1,len(instances)):
                try:
                    node_A = node_mapping[instances[i].inst.name]
                    node_B = node_mapping[instances[j].inst.name]
                    edge_index.append([node_A, node_B])
                    edge_index.append([node_B, node_A])
                except:
                    pass
    

    print(edge_index)
    # print(len(edge_index))
    # return edge_index

# node 0 will be M1
# node 1 will be M2
# node 2 will be M3
# node 3 will be M4
# node 4 will be M5
# node 5 will be Mp
# node 6 will be Vb
# node 7 will be Vdd
# node 8 will be GND
# node 9 will be Rfb
# node 10 will be Cfb
# node 11 will be Cdec
get_net_edge_index(dict(M1 = 0,
                        M2 = 1,
                        M3 = 2,
                        M4 = 3,
                        M5 = 4,
                        Mp = 5,
                        Vb = 6,
                        Vdd = 7,
                        I2 = 8,
                        Rfb = 9,
                        Cfb = 10,
                        Cdec = 11),
                   lib_cell_name = "Low_Dropout_With_Diff_Pair")


# edit_schematic('M1', dict(l = '123u', 
#                           w = '456u',
#                           simM = '13'))

# edit_schematic('V1', dict(vdc = '123u', 
#                           acm = '456u'))



