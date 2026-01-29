#!/usr/bin/env python3
"""
Generate graph visualization (graph.png) from the LangGraph workflow.
Run this script to create/update the workflow diagram.
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the compiled graph
from graph.graph import app

# Generate the visualization
print("Generating graph visualization...")
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("✓ Graph visualization saved to graph.png")
