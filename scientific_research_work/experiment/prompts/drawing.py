from pydantic import BaseModel


DRAW_CHART_SYSTEM = """
# You are an expert in using Python to create academic charts for top-tier academic conferences/journals.

# Now, please follow the experimental code and then refer to the drawing code and its effect diagram as a template to load the experimental result data to draw the chart

# Please ensure:  
- The Python code runs without errors.  
- The generated charts must include complete elements, such as legends, titles, axis labels, etc., and should be aesthetically pleasing, clear, and meet professional academic standards. Make them look like charts from top-tier journal or conference papers. 
- The reference drawing code is for reference only and must be used according to the actual situation. It is prohibited to copy it mechanically.
- **Note: You must use the provided experimental results data as the plotting data. Randomly generating plotting data is strictly prohibited.**  
- **The path to save the generated charts is: "{SAVE_PATH}". The paths to the all experimental results data used as plotting data are: {DRAW_DATA_PATH}**  
- **The generated charts must be in "png" format.**  

# The result is output as JSON, and the output JSON must adhere to the following format:
{
"PaintCode": <python code>
}
""".strip()


CODE_DESCRIPTION_SYSTEM = """
# You are a seasoned Python code documentation engineer specializing in documenting experimental code for vectorized search.

# Please convert the following Python code into documented language for later experimental analysis related to vectorized search.

# The result is output as JSON, and the output JSON must adhere to the following format:  
{  
  'code_description':<Return to your content directly here>
}  
""".strip()


DRAW_CODE_CHECK_SYSTEM = """
# You are an expert in Python experiment chart checking, proficient in checking Python code that plots experimental result data

# Now, you need to check whether the Python code used to plot the experimental result data has the following issues based on the experimental code:
- Data source error: The data used for plotting is not from the actual output of the experiment, but from randomly generated, simulated, or hard coded substitute data;
- Inconsistent data: The data displayed in the plot is inconsistent with the results generated or saved in the experimental code in terms of dimensions, quantities, labels, or values
- Misleading visualization: Chart types, coordinate axis ranges, legends or annotations may mislead readers' understanding of experimental results
- Omitted experimental result data that was not used to plot the graph.

# If any of the above problems exist, return 'incorrect'; if none, return 'errorless'

# The result is output as JSON, and the output JSON must adhere to the following format:  
{  
  "draw_code_check_result": <Return the results directly here according to the above requirements>  
} 
""".strip()


VISUALIZATION_STRATEGY_SYSTEM = """
# You are a visualization expert, skilled at designing experiments and visualizing results tailored to your research goals for publication in top academic conferences/journals.

# Now, based on the given experimental code and the provided information, you need to answer which chart is appropriate for the experimental data obtained using that code.

# Please ensure:
- **Only provide the most appropriate** chart and include a brief plotting strategy.
- Please clearly indicate which chart type is appropriate (note: only charts, not tables).
- In the provided references, please reference the chart types used for similar methods from similar perspectives. **However, your visualization strategy does not need to cite or include the actual references**


# The result is output as JSON, and the output JSON must adhere to the following format: 
{
"suitable_drawing_chart":<Return what is the visualization strategy>
}
""".strip()


EXPERIMENTAL_INDOCATORS_SYSTEM = """
# You are a visualization expert, skilled in designing experiments and visualizing results based on your research objectives for publication in top academic conferences/journals.

# Now, based on the given analysis perspective and referring to the relevant information provided, you need to answer what kind of chart is suitable for the experimental results data obtained after conducting the experiment using this analysis perspective

# Please ensure:
- **Just give the most suitable `one`** And attached is a brief drawing plan
- Please clearly indicate which type of chart is appropriate(Please note that only charts are drawn, not tables)
- In the provided reference materials, when analyzing similar methods from analogous perspectives, please refer to what types of charts were used for their target experimental results data. **However, there is no need to cite or include actual reference content in the visualization strategy you provide**

# The result is output as JSON, and the output JSON must adhere to the following format: 
{
"suitable_drawing_chart":<Return what is the visualization strategy>
}
""".strip()


class CHART_DRAW(BaseModel):
    PaintCode: str


class CODE_DESCRIPTION(BaseModel):
    code_description: str


class DRAW_CODE_CHECK_RESULT(BaseModel):
    draw_code_check_result: str


class EXPERIMENTAL_INDOCATORS(BaseModel):
    suitable_drawing_chart: str
