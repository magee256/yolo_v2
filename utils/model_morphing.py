"""
Defines utilities that allow updating a model's architecture
"""

from keras.models import Model, clone_model
from keras.layers import InputLayer


def sub_in_layer(model, sub_out_name, sub_in_layer):
    """
    Replaces a node with an input layer and updates the model graph to reflect that
    change. Unable to handle layers with mulitple inbound_nodes. 
    """
    head_dict = {sub_out_name: sub_in_layer.output}  # Assumes one output
    sub_out_layer = model.get_layer(sub_out_name)
    out_node_list = [node for node in sub_out_layer.outbound_nodes]
    continue_loop = True
    while out_node_list and continue_loop:
        continue_loop = False
        for node in out_node_list[::-1]:
            modified = update_graph(model, head_dict, node, out_node_list)
            continue_loop = continue_loop | modified
        if not continue_loop:
            raise ValueError('Could not find reference to inbound layer. This'
                ' likely means output from a layer before the subbed out layer'
                ' is needed.')
    

def update_graph(model, head_dict, out_node, out_node_list):
    """Update graph layers to take output from tensors specified in head_dict"""
    if input_ref_present(out_node, head_dict):
        new_out_nodes = relink_graph(model, out_node, head_dict)
        out_node_list.remove(out_node)
        new_out_nodes = [node for node in new_out_nodes if node not in out_node_list]
        out_node_list += new_out_nodes
        return True
    return False


def input_ref_present(out_node, head_dict):
    return all(layer_name in head_dict.keys() 
               for layer_name in out_node.get_config()['inbound_layers'])
                    

def relink_graph(model, out_node, head_dict):
    """
    Feed output of inbound_layers specified in out_node to the receiving layer. 
    Associate the receiving layer's name with its new output in head_dict. 
    Return the receiving layer's outbound nodes to continue relinking graph. 
    """
    new_link_name = out_node.get_config()['outbound_layer']
    new_link_layer = model.get_layer(new_link_name)
    if len(new_link_layer.inbound_nodes) > 1:
        raise NotImplementedError('Layer {} used in multiple places'.format(new_link_name)
                                  + ' (has multiple inbound_nodes). Unable to relink graph.')
    
    if len(out_node.get_config()['inbound_layers']) == 1:
        new_link_layer.inbound_nodes = []
        x = new_link_layer(head_dict[out_node.get_config()['inbound_layers'][0]])
    else:
        new_link_layer.inbound_nodes = []
        inbound_list = [head_dict[l] for l in out_node.get_config()['inbound_layers']]
        x = new_link_layer(inbound_list)
    head_dict[new_link_name] = x
    return new_link_layer.outbound_nodes


def model_top_at_layer(model, layer_name):
    """
    Returns the part of model from layer layer_name until the output layer
    with layer layer_name replaced by an InputLayer
    """
    mod_cop = clone_model(model)
    mod_cop.set_weights(model.get_weights())
    
    inp = InputLayer(model.get_layer(layer_name).output_shape[1:])
    sub_in_layer(mod_cop, layer_name, inp)
    x = mod_cop.layers[-1].get_output_at(1)
    
    return Model(inp.input, x)
