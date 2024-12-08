#!/usr/bin/env python

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

# Load ONNX models
model1 = onnx.load("model1.onnx")
model2 = onnx.load("model2.onnx")

# Add prefixes to avoid naming conflicts
add_prefix_to_graph(model1.graph, "m1_")
add_prefix_to_graph(model2.graph, "m2_")

# Unify input: Use the input of model1 for both models
unified_input = model1.graph.input[0]
combined_inputs = [unified_input]

# Update the inputs of the second model's nodes to use the first model's input
for node in model2.graph.node:
    node.input[:] = [
        inp.replace("m2_", "m1_") if inp in [i.name for i in model2.graph.input] else inp
        for inp in node.input
    ]

# Combine nodes and initializers
combined_nodes = list(model1.graph.node) + list(model2.graph.node)
combined_initializers = list(model1.graph.initializer) + list(model2.graph.initializer)

# Outputs of both models
classification_output = model1.graph.output[0].name
regression_output = model2.graph.output[0].name

# Add a concatenation node for final output
concat_output_name = "final_output"
concat_node = helper.make_node(
    "Concat",
    inputs=[classification_output, regression_output],
    outputs=[concat_output_name],
    axis=1,  # Concatenate along the feature axis
    name="ConcatOutputs"
)
combined_nodes.append(concat_node)

# Define the combined output
combined_outputs = [
    helper.make_tensor_value_info(concat_output_name, onnx.TensorProto.FLOAT, [None, None])
]

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

# Save the combined model
onnx.save(combined_model, "combined_model.onnx")

print("Merged model saved as 'combined_model.onnx'")
