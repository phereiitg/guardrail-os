"""
One-off: render the compiled LangGraph pipeline to a PNG.
Run once from the project root:  python make_graph_image.py
Produces pipeline_graph.png in the project root.

This imports the already-built `pipeline` from app.main, so the server must NOT
be running at the same time (both would load the models and contend for the GPU).
"""

from app.main import pipeline

png = pipeline.get_graph().draw_mermaid_png()
with open("pipeline_graph.png", "wb") as f:
    f.write(png)

print("Saved pipeline_graph.png")
print("\nMermaid source (in case the PNG renderer needs internet / fails):\n")
print(pipeline.get_graph().draw_mermaid())