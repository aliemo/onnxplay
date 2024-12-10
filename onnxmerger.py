#!/usr/bin/env python

import argparse
import onnx
from onnx import helper

def add_prefix_to_graph(graph, prefix):
    """
    Adds a prefix to inputs, outputs, nodes, and initializers in an ONNX graph.
    """
    # Add prefix to inputs
    for input_tensor in graph.input:
        input_tensor.name = prefix + input_tensor.name

    # Add prefix to outputs
    for output_tensor in graph.output:
        output_tensor.name = prefix + output_tensor.name

    # Add prefix to nodes
    for node in graph.node:
        node.name = prefix + node.name
        node.input[:] = [prefix + inp for inp in node.input]
        node.output[:] = [prefix + out for out in node.output]

    # Add prefix to initializers
    for initializer in graph.initializer:
        initializer.name = prefix + initializer.name

    return graph


def main(args):

    # Load ONNX models
    
    path1 = args.model1
    path2 = args.model2
    
    prefix1 = args.prefix1
    prefix2 = args.prefix2
    
    input_name1 = args.input1
    input_name2 = args.input2
    output_name = args.output
    
    path = args.path

    model1 = onnx.load(path1)
    if args.verbose:
        print(f'read model 1 from {path1}')

    model2 = onnx.load(path2)
    if args.verbose:
        print(f'read model 2 from {path1}')
    
    # Add prefixes to avoid naming conflicts
    if args.verbose:
        print(f'add  model1 prefix: {prefix1}')
    add_prefix_to_graph(model1.graph, prefix1)

    if args.verbose:
        print(f'add  model2 prefix: {prefix2}')
        add_prefix_to_graph(model2.graph, prefix2)

    
    
    # Unify input: Use the input of model1 for both models
    unified_input = model1.graph.input[0]
    combined_inputs = [unified_input]
    if args.verbose:
        print(f'models inputs are unified: {combined_inputs}')
        
    # Update the inputs of the second model's nodes to use the first model's input
    for node in model2.graph.node:
        node.input[:] = [
            inp.replace(prefix2, prefix1) if inp in [i.name for i in model2.graph.input] else inp
            for inp in node.input
        ]
    if args.verbose:
        print(f'reconstruct combined model')
    # Combine nodes and initializers
    combined_nodes = list(model1.graph.node) + list(model2.graph.node)
    
    combined_initializers = list(model1.graph.initializer) + list(model2.graph.initializer)

    
    # Outputs of both models
    output1 = model1.graph.output[0].name
    output2 = model2.graph.output[0].name


    # Add a concatenation node for final output
    concat_output_name = output_name
    concat_node = helper.make_node(
        "Concat",
        inputs=[output1, output2],
        outputs=[concat_output_name],
        axis=1,  # Concatenate along the feature axis
        name="ConcatOutputs"
    )
    combined_nodes.append(concat_node)

    # Define the combined output
    combined_outputs = [
        helper.make_tensor_value_info(concat_output_name, onnx.TensorProto.FLOAT, [None, None])
    ]
    if args.verbose:
        print(f'concat outputs of models')
    # Create the combined graph
    combined_graph = helper.make_graph(
        nodes=combined_nodes,
        name="CombinedGraph",
        inputs=combined_inputs,
        outputs=combined_outputs,
        initializer=combined_initializers,
    )


    # Create the combined model
    combined_model = helper.make_model(combined_graph, producer_name="MergeONNX")
    if args.verbose:
        print(f'merge outputs of models')
    
    if args.verbose:
        print(f'saving final model...')    
    
    onnx.save(combined_model, path)

    print(f"model saved as {path}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="A simple argparse example script."
    )
    
    
    parser.add_argument("--model1", type=str, help="onnx model 1 path", default='model1.onnx')
    parser.add_argument("--model2", type=str, help="onnx model 2 path", default='model2.onnx')
    parser.add_argument("--prefix1", type=str, help="onnx model 1 prefix", default='m1_')
    parser.add_argument("--prefix2", type=str, help="onnx model 2 prefix", default='m2_')
    parser.add_argument("--input1", type=str, help="onnx model 1 input name", default='input')
    parser.add_argument("--input2", type=str, help="onnx model 2 input name", default='input')
    parser.add_argument("--output1", type=str, help="onnx model 1 output name", default='reserveed')
    parser.add_argument("--output2", type=str, help="onnx model 2 output name", default='reserveed')
    parser.add_argument("--input", type=str, help="onnx final model input name", default='fin')
    parser.add_argument("--output", type=str, help="onnx final model output name", default='fout')
    parser.add_argument("--path", type=str, help="onnx final model output savepath")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    # Parse the arguments
    args = parser.parse_args()
    
    main(args)
