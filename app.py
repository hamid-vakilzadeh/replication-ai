import os
import toml

# Load the configuration file
config = toml.load('.secret.toml')


os.environ["SERPER_API_KEY"] = config['serper']
os.environ["OPENAI_API_KEY"] = config['openai']
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

from crewai import Agent
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI

pdf_search_tool = PDFSearchTool(pdf='Jones-EarningsManagementImport-1991.pdf')

llm = ChatOpenAI(model='gpt-4o', temperature=0.3)

# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Award Winning Accounting Academic Researcher',
  goal='Fully Layout the step to calculate {measure} in markdown format.',
  verbose=True,
  memory=True,
  backstory=(
    "You are a seasoned researcher with a keen eye for detail."
    "You have a knack for uncovering the details of the research design"
    "and methodology of academic papers, and you are able to"
    "layout the steps to calculate various measures used in research."
  ),
  tools=[pdf_search_tool],
  allow_delegation=True,
  llm=llm
)

# Creating a writer agent with custom tools and delegation capability
analyst = Agent(
  role='Award Winning Academic Data Analyst',
  goal='List a details review of necessary data fields to calculate {measure} in markdown format.',
  verbose=True,
  memory=True,
  backstory=(
    "You are a data analyst with a passion for academic research."
    "You have a deep understanding of the variables used in research papers,"
    "and you are able to list the fields necessary to calculate various measures."
    "When using the PDF search tool, instead of single words, use elaborate sentences."
  ),
  tools=[pdf_search_tool],
  allow_delegation=False,
  llm=llm
)


from crewai import Task

# Research task
research_task = Task(
  description=(
    "find the description of the {measure} calculation. "
    "Provide a detailed step by step description of the {measure} calculation."
  ),
  expected_output='A comprehensive step by step description of the {measure} calculation, '
                  'Including all the details necessary to calculate the {measure}.'
                  ' in markdown format.',
  tools=[pdf_search_tool],
  agent=researcher,
  output_file='research_design.md'  # Example of output customization
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Write all the fields necessary to calculate the {measure} in the paper."
    "Provide a bullet list of all the fields used in the paper to calculate the {measure} with"
    "a comprehensive description of each field. "
    "Also provide the database where the data was extracted from."
  ),
  expected_output='A bullet list of all the variables used in the paper in markdown format.',
  tools=[pdf_search_tool],
  agent=analyst,
  async_execution=False,
  output_file='variable_list.md'  # Example of output customization
)

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[researcher, analyst],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=20,
  share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff({'measure': 'Discretionary Accruals'})
print(result)
