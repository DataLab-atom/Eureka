from pydantic import BaseModel


CHART_DESCRIPTION_SYSTEM = """
# You are an expert in chart description, please generate a chart description for the user-provided chart. The chart description will be embedded for data retrieval.

# Please ensure:
- Only output the descriptive text.
- The chart description needs to be clear and comprehensive, detailing all aspects of the chart.
- Do not add any explanations, titles, or additional content.
- Do not alter the original meaning of the content.

# The result is output as JSON, and the output JSON must adhere to the following format:
{
"chart_description": <Go directly to the description of the diagram you wrote>
}
""".strip()


class ChartDescription(BaseModel):
    chart_description: str
