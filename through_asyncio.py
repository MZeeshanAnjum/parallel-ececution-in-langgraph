from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llm import llm, get_custom_llm
from typing import List,Dict, Union, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from agent_flow.caio.prompts.prompt import AGENT_INFORMATION_RESPONDER1, AGENT_INFORMATION_RESPONDER2, CAIO_PROMPT, Report_generation_prompt,Scrum_CAIO_PROMPT, SCHEDULED_REPORT_GENERATION, MEMORY_DECISION_PROMPT, INTENT_CLASSIFIER_PROMPT, FINAL_RESPONSE_PROMPT, SIMPLE_RESPONSE_PROMPT, GENERATE_SEMANTIC_QUERY,handle_RAG_response, CREATE_FETCH_COMMAND
from agent_flow.caio.utils.logger import get_custom_logger
from db_helping_tools.db_functions import insert_or_update_kpis, get_agent_names_by_position, update_agent_response, get_agent_card_by_agent_name, get_agent_response, get_agent_kpi, get_buisness_information, get_sub_agents,hybrid_search
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import tool
from agent_flow.requirements_agent.utils.rag import create_and_store_embeddings
from agent_flow.prompt_refiner.agent import get_refine_agent_tool

import json
from agent_flow.caio.utils.remote_agent import call_remote_agent, scheduled_invoke_CAIO_pipeline
from langgraph.types import Command

import asyncio
from datetime import datetime
from datetime import timedelta
from config import get_db_connection

logger = get_custom_logger("CAIO")


def override(_, new):
    return new

class TeamInformation(BaseModel):
    query: str = ""
    flag: bool = False

class RagResponse(BaseModel):
    query:str = ""
    top_k :int =3
    flag: bool = False

class FollowUpQuestion(BaseModel):
    follow_up_question: str =""

class GeneralResponse(BaseModel):
    simple_response: Optional[bool] = None
    follow_up_question : Optional[FollowUpQuestion] = None


class Intents(BaseModel):

    team_information: Optional[TeamInformation] = None
    call_on_demand : bool = False
    business_rag_response: Optional[RagResponse] = None
    agent_analysis: bool = False
    simple_response : GeneralResponse = GeneralResponse()
    call_managers : bool = False


class state(BaseModel):
    Q_A : bool =True
    has_set_kpis: bool = False
    query: str = ""
    agent_names: List[str] = []
    manager_names: List[str] = []
    messages: Annotated[list[dict], override] = []
    call: bool = False
    has_run_pipelines: bool = False
    to_call: Dict[str, str] = {}
    agent_cards: List[dict] = []
    hierarchy: Dict[str, List[str]] = {}
    user_instruction: bool = False
    user_intents : Intents = Intents()
    retrieved_text: str = ""
    caio_response: str = ""
    business_info: str = ""
    user_id: int = 1
    internal_messages: Annotated[list[dict], override] = []
    parallel_intents:bool = False

class ResponseWithFlag(BaseModel):
    Status: bool
    response: str

class BooleanResponse(BaseModel):
    flag: bool

class ManagerNames(BaseModel):
    manager_name : str
    query : str

class CaioResponse(BaseModel):
    response : List[ManagerNames]

class AgentResponseQuery(BaseModel):
    """
    Schema for fetching agent responses intelligently
    based on user intent.
    """

    agent_names: Union[str, List[str]] = Field(
        ...,
        description="Single agent name or a list of agent names to fetch responses for."
    )

    fetch_type: Literal["latest", "time_range"] = Field(
        "latest",
        description="Determines whether to fetch the latest response(s) or responses over a time range."
    )

    start_time: Optional[datetime] = Field(
        None,
        description="Start timestamp (UTC) for time range queries."
    )

    end_time: Optional[datetime] = Field(
        None,
        description="End timestamp (UTC) for time range queries."
    )
    
    limit: Optional[int] = Field(
        1,
        description="Maximum number of responses to fetch per agent for 'latest' fetch type."
    )


class CAIO_Agent():

    def __init__(self):

        self.graph = self.build_graph()

    def update_messages(self,state: state) -> state:

        user_id = state.user_id
        conn = get_db_connection()
        cur = conn.cursor()
        response = cur.execute(f'SELECT agent_name FROM "Team" WHERE user_id={user_id};')
        data = cur.fetchall()

        hierarchy = {}
        agent_names = [item[0] for item in data]
        manager_names = get_agent_names_by_position("Manager",user_id=user_id)

        logger.info("IN THE UPDATE MESSAGES NODE \n")

        messages = state.messages.copy()
        user_query = state.query

        messages.append({"role": "user", "content": user_query})
        return {"messages": messages,"internal_messages": [],"agent_names":agent_names, "manager_names": manager_names}

    async def get_kpis(self, state: state) -> state:

        logger.info("IN THE GET TEAM NODE \n")
        messages = state.messages.copy()
        agent_names = state.agent_names

        agent_cards = {}
        for agent in agent_names:
            agent_card = get_agent_card_by_agent_name(agent, user_id=state.user_id)
            agent_cards[agent] = agent_card

        formatted_agent_cards = "\n".join([
            f"{agent}: description: {card.get('description', 'No description available')} \t tools: {card.get('tools', [])} \t KPI set by user: {card.get('kpi', '')}"
            for agent, card in agent_cards.items() if card
        ])

        print(formatted_agent_cards)
        formatted_context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])

        if state.parallel_intents:
            prompt = AGENT_INFORMATION_RESPONDER1.format(agent_cards=formatted_agent_cards, query= state.user_intents.team_information.query if state.user_intents.team_information.flag else "", context_history=formatted_context_history)
            ai = [AIMessage(content=prompt)]
            tools= [insert_or_update_kpis,get_refine_agent_tool(user_id=state.user_id)]
            llm_agent = get_custom_llm(temperature =0.4)
            llm_with_tools = llm_agent.bind_tools(tools)
            llm_response = await llm_with_tools.ainvoke(ai)
            logger.info(llm_response)

        else:

            prompt = AGENT_INFORMATION_RESPONDER2.format(agent_cards=formatted_agent_cards, context_history=formatted_context_history)
            ai = [AIMessage(content=prompt)]
            tools= [insert_or_update_kpis,get_refine_agent_tool(user_id=state.user_id)]
            llm_with_tools = llm.bind_tools(tools)
            llm_response = await llm_with_tools.ainvoke(ai)
            logger.info(llm_response)

        if "tool_calls" in llm_response.additional_kwargs:

            message = {
                "role": "assistant",
                "content": llm_response.content,
                "additional_kwargs": getattr(llm_response, "additional_kwargs", {}),
                "sender": "get_kpis",
            }

            msg = self.tool_execution_node(message, state.user_id, tools)
            msg = {"role": "Team Assistant", "content": msg}

        else:

            msg = {"role": "Team Assistant", "content": llm_response.content}

        if state.parallel_intents:
            # FIX: Append to existing internal_messages
            internal_messages = state.internal_messages.copy()
            internal_messages.append(msg)
            return {"internal_messages": internal_messages}
            
        else:
            return {"messages": messages + [msg]}
            # print(" \n \n Sending to report_generation \n \n ")
            # return Command(
            #     update={"internal_messages": [msg]},
            #     goto = "report_generation"
            # )


    def intent_classifier(self, state: state) -> state:
        logger.info("IN THE INTENT CLASSIFIER NODE \n")
        messages = state.messages.copy()
        context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-8:]])
        prompt = INTENT_CLASSIFIER_PROMPT.format(context_history=context_history)

        ai = [AIMessage(content=prompt)]
        llm_0 = get_custom_llm(temperature =0.1)
        structured_llm = llm_0.with_structured_output(Intents)
        llm_response = structured_llm.invoke(ai)

        logger.info(llm_response)

        if isinstance(llm_response, Intents):
            llm_response = llm_response

        if isinstance(llm_response, dict):
            llm_response = Intents(**llm_response)

        parallel_intents=[]

        # Check for multiple intents that can run in parallel


        if llm_response.business_rag_response:
            parallel_intents.append("business_rag_response")
        
        if llm_response.agent_analysis:
            parallel_intents.append("agent_analysis")

        if llm_response.team_information:
            parallel_intents.append("team_information")

        if llm_response.call_managers:
            parallel_intents.append("call_managers")

        if len(parallel_intents) > 1:
            logger.info(f"Parallel intents detected: {parallel_intents}")
            parallel_intents = True

        else:
            parallel_intents = False
        return {"user_intents": llm_response, "parallel_intents": parallel_intents}

    async def CAIO(self, state: state) -> state:

        logger.info("IN THE CAIO \n")
        messages = state.messages.copy()
        user_id = state.user_id

        hierarchy = {}

        manager_names = state.manager_names
        # agent_cards = [get_agent_card_by_agent_name(name, user_id=user_id) for name in manager_names]

        for agent in manager_names:
            agent_card = get_agent_card_by_agent_name(agent, user_id=user_id)
            # hierarchy[agent] = get_sub_agents(agent, user_id=user_id)
            hierarchy[agent] = {"description": agent_card.get("description", "No description available")}
        logger.info(f"Hierarchy: {hierarchy}")

        context_history = state.messages
        internal_messages = state.internal_messages

        formatted_agent_cards = "\n\n".join([
            f"**Manager Agenr Name:** {agent}\n**Description:** {info['description']}"
            for agent, info in hierarchy.items()
        ])

        logger.info(f"formatted_agent_cards: {formatted_agent_cards}")
        formatted_context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_history[-6:]])
        prompt = CAIO_PROMPT.format(manager_names=manager_names, context_history=formatted_context_history, agent_cards=formatted_agent_cards)

        ai = [AIMessage(content=prompt)]
        caio_llm = get_custom_llm(temperature = 0.4)
        structured_llm = caio_llm.with_structured_output(CaioResponse)
        llm_response = await structured_llm.ainvoke(ai)
        print(llm_response.response)
        logger.info(f"Type: {type(llm_response.response)}")

        if state.user_intents.call_on_demand:
            logger.info("INVOKING ON DEMAND MANAGER AGENTS")
            messages = state.messages.copy()  # make a local copy

            async def handle_item(item):
                logger.info(f"Calling Manager : {item.manager_name} with query: {item.query}")
                remote_response = await call_remote_agent(item.manager_name, item.query, user_id=state.user_id)
                update_agent_response(item.manager_name, remote_response, user_id=state.user_id )

                internal_messages.append({
                    "role": item.manager_name,
                    "content": remote_response,
                    "source": "This is a live response from agent",
                })

            # Run all manager calls in parallel
            tasks = [handle_item(item) for item in llm_response.response]
            await asyncio.gather(*tasks)

            # After all tasks complete, proceed to report generation
            return {"messages": messages, "internal_messages": internal_messages}

        elif state.user_intents.call_managers:
            logger.info("FETCHING RESPONSES FROM DB FOR MANAGER AGENTS")
            
            for item in llm_response.response:
                logger.info(f"Getting latest response from Manager : {item.manager_name}")
                response, updated_at = get_agent_response(item.manager_name, user_id=state.user_id)

                internal_messages.append({
                    "role": item.manager_name,
                    "content": response,
                    "source": "**The response is fetched from DB**",
                    "updated_at": updated_at,
                })

            return {"messages": messages, "internal_messages": internal_messages}

    def report_generation(self, state: state) -> state:
        """
        Custom node to handle report generation from agent responses.
        It compiles the information from all agents into a structured report.
        """

        logger.info("\n IN REPORT GENERATION NODE FOR Q_A \n")
        messages = state.messages.copy()
        context_history = messages[-6:] + state.internal_messages
        context_history = "\n".join([
            (
                f"User: {msg['content']}"
                if msg["role"].lower() == "user"
                else f"Agent name: {msg['role']} | Source: {msg.get('source', '')} | Updated At: {msg.get('updated_at', '')}\n{msg['content']}"
            )
            for msg in context_history
        ])
        
        print(f"\n {context_history} \n")

        prompt = Report_generation_prompt.format(user_query=state.query, context_history=context_history,)

        ai = [AIMessage(content=prompt)]
        llm_response = llm.invoke(ai)
        logger.info({llm_response.content})

        messages.append({
            "role": "assistant",
            "content": llm_response.content,
        })

        return {"caio_response": llm_response.content, "messages": messages}
    

    async def handle_simple_response(self, state: state) -> state:

        internal_messages = state.internal_messages.copy()

        messages = state.messages.copy()
        if messages:
            message_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])
        else:
            message_history = "No prior conversation yet."
        # message_history = "\n".join([f"{msg['role']}: {msg['content']}"] for msg in messages[-8:] if messages else "No prior conversation yet.")
        intents = state.user_intents


        if intents.simple_response.simple_response:

            logger.info("IN SIMPLE RESPONSE")
            follow_up_question = ""
            if intents.simple_response.follow_up_question:
                follow_up_question = intents.simple_response.follow_up_question

            logger.info("SIMPLE MESSAGE IS TRUE \n")
            prompt = SIMPLE_RESPONSE_PROMPT.format(user_query=state.query, context_history=message_history, follow_up_question=follow_up_question)
            ai = [AIMessage(content=prompt)]
            llm_response = await llm.ainvoke(ai)

            logger.info(f" \n {llm_response.content} \n")
            messages.append({
                "role": "assistant",
                "content": llm_response.content
            })
            internal_messages.append({
                "role": "RAG Assistant",
                "content": llm_response.content
            })

            return {"internal_messages": internal_messages, "messages": messages}

        if intents.business_rag_response.flag:

            logger.info("\n BUSINESS RAG IS TRUE \n")

            retrieved_text = self.rag_agent(state)

            if state.parallel_intents:
                msg = {
                    "role": "RAG Assistant",
                    "content": f"Retrieved text for RAG: {retrieved_text}"
                }
                return {"internal_messages": [msg]}
            else:

                context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-8:] ])
                prompt = handle_RAG_response.format(user_query=state.query, context_history=context_history, retrieved_text=retrieved_text)

                ai = [AIMessage(content=prompt)]
                llm_response = await llm.ainvoke(ai)
                logger.info(f" \n {llm_response.content} \n")

                # gathered_info["rag"] = llm_response.content
                messages.append({
                    "role": "assistant",
                    "content": llm_response.content,
                })

                internal_messages.append({
                    "role": "RAG Assistant",
                    "content": llm_response.content
                })

                return {"messages": messages, "internal_messages": internal_messages}

    async def business_update(self, state: state) -> state:
        logger.info("IN THE BUSINESS UPDATE NODE \n")
        messages = state.messages.copy()
        context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-4:]])

        prompt = MEMORY_DECISION_PROMPT.format(user_query = state.query, conversation_history=context_history)
        structured_llm = llm.with_structured_output(BooleanResponse)

        ai = [SystemMessage(content=prompt)]
        llm_response = await structured_llm.ainvoke(ai)

        logger.info(f"Response for business update: {llm_response.flag}")

        return {"user_instruction": llm_response.flag}


    def rag_agent(self, state: state) -> state:

        conversation_history = state.messages.copy()
        query = state.query
        context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]])

        if state.user_intents.business_rag_response:
            logger.info(f" \n GENERATING BUSINESS RAG QUERY for {state.query} \n")

            prompt = GENERATE_SEMANTIC_QUERY.format(user_query=query, context_history=context_history, current_datetime= datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

        ai = [AIMessage(content=prompt)]
        structured_llm = llm.with_structured_output(RagResponse)
        llm_response = structured_llm.invoke(ai)
        logger.info(f"Re-fabricated query for RAG: {llm_response}")
        top_k_1 = min(llm_response.top_k, 7)

        retrieved_text_1 = hybrid_search(llm_response.query, top_k=top_k_1, user_id=state.user_id)
        retrieved_text_2 = hybrid_search(state.user_intents.business_rag_response.query, top_k=3, user_id=state.user_id)

        print(f" \n \n Retrieved text for Intent Question {top_k_1}: {retrieved_text_1} \n \n")
        print(f" \n \n Retrieved text for Background Information {state.user_intents.business_rag_response.top_k}: {retrieved_text_2} \n \n")

        # prompt = RAG_FILTER_PROMPT.format(user_query=user_query, retrieved_text=retrieved_text, context_history=context_history)

        # ai = [SystemMessage(content=prompt)]
        # structured_llm = llm.with_structured_output(ResponseWithFlag)
        # llm_response = structured_llm.invoke(ai)
        # print(f"\n{llm_response} \n")

        return retrieved_text_1 + "\n" + retrieved_text_2
    
    async def agent_response_analysis(self, state: state) -> state:

        logger.info("IN THE AGENT RESPONSE ANALYSIS NODE \n")
        messages = state.messages.copy()
        context_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])

        hierarchy = {}

        agent_names = state.agent_names

        prompt = CREATE_FETCH_COMMAND.format(context_history=context_history, team_hierarchy=agent_names, current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        llm_agent_response = get_custom_llm(temperature = 0.4)
        structured_llm = llm.with_structured_output(AgentResponseQuery)

        ai = [SystemMessage(content=prompt)]
        llm_response = structured_llm.invoke(ai)

        logger.info(f"Response for agent analysis : {llm_response}")
        agent_responses = self.execute_agent_query(llm_response, user_id=state.user_id)

        logger.info(f"Fetched agent responses: \n {agent_responses}")

        # Process the agent responses as needed
        msgs = []

        for agent_response in agent_responses:
            response = {"role": agent_response.get("agent_name", "unknown"), "content": agent_response.get("response", ""), "source": "**The response is fetched from DB**", "updated_at": agent_response.get("updated_at", "unknown time")}
            msgs.append(response)

        if state.parallel_intents:

            return {"internal_messages": msgs}

        else:
            return Command(
                update={"internal_messages": msgs},
                goto = "report_generation"
            )

    def execute_agent_query(self, params: AgentResponseQuery, user_id: int):
        """
        Execute agent query using Supabase client
        """
        if isinstance(params.agent_names, str):
            agent_names = [name.strip() for name in params.agent_names.split(",") if name.strip()]
        else:
            agent_names = params.agent_names or []

        if not agent_names:
            logger.warning("No agent names provided to fetch_agent_responses.")
            return []

        fetch_type = params.fetch_type or "latest"
        limit = params.limit or 1
        results = []

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            if fetch_type == "time_range" and params.start_time and params.end_time:
                # --- Time range query ---
                end_time = params.end_time + timedelta(days=1)
                query = """
                    SELECT agent_name, response, updated_at
                    FROM public."AgentResponses"
                    WHERE user_id = %s
                    AND agent_name = ANY(%s)
                    AND updated_at >= %s AND updated_at < %s
                    ORDER BY updated_at DESC;
                """
                cur.execute(query, (user_id, agent_names, params.start_time, end_time))
                rows = cur.fetchall()
                results = [
                    {"agent_name": row[0], "response": row[1], "updated_at": row[2]}
                    for row in rows
                ]

            else:
                # --- Latest responses per agent ---
                for agent_name in agent_names:
                    cur.execute(
                        """
                        SELECT agent_name, response, updated_at
                        FROM public."AgentResponses"
                        WHERE user_id = %s AND agent_name = %s
                        ORDER BY updated_at DESC
                        LIMIT %s;
                        """,
                        (user_id, agent_name, limit),
                    )
                    rows = cur.fetchall()
                    results.extend(
                        [
                            {"agent_name": row[0], "response": row[1], "updated_at": row[2]}
                            for row in rows
                        ]
                    )

            logger.info(f"Fetched {len(results)} responses for agents: {agent_names}")
            return results

        except Exception as e:
            logger.error(f"Error executing agent query: {e}")
            return []

    async def process_multiple_intents(self, state: state) -> state:
        """Process multiple intents in parallel and populate internal_messages"""
        
        # Create tasks for each intent that's True
        tasks = []
    

        if state.user_intents.business_rag_response:
            # tasks.append(self.handle_simple_response(state))
            tasks.append(self.handle_simple_response(state))

        if state.user_intents.agent_analysis:
            # tasks.append(self.agent_response_analysis(state))
            tasks.append(self.agent_response_analysis(state))

        if state.user_intents.call_managers:
            tasks.append(self.CAIO(state))

        if state.user_intents.team_information.flag:
            tasks.append(self.get_kpis(state))

        # Run all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Populate internal_messages with results

            internal_messages = []

            for result in results:
                if not isinstance(result, Exception) and result:
                    # Only extract the "internal_messages" key if it exists and is a list
                    if isinstance(result, dict) and "internal_messages" in result:
                        internal_messages.extend(result["internal_messages"])


            logger.info(f"Total internal messages from parallel intents: {len(internal_messages)}")
            logger.info("=" * 60)
            logger.info(f"Total messages: \n  {internal_messages} \n \n")
                    
        return {"internal_messages": internal_messages}

    def tool_execution_node(self, message,user_id, tools) -> str:
        """
        Custom node to handle tool execution when LLM requests it.
        It extracts the tool call from the latest AIMessage and executes the tool.
        """
        if not message:
            return ""  # nothing to do


        # Check if last message has a tool call
        tool_calls = message["additional_kwargs"].get("tool_calls", [])
        
        # Check if last message has a tool call
        # tool_calls = last_msg.get("additional_kwargs", {}).get("tool_calls", [])
        if tool_calls:
            print(f"Tool calls found: {tool_calls}")
        
            for tool_call in tool_calls:

                # Extract properly
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name")
                raw_args = function_data.get("arguments", "{}")

                try:
                    tool_args = json.loads(raw_args)  # convert JSON string to dict
                    tool_args["user_id"] = user_id
                except Exception as e:
                    logger.error(f"Failed to parse tool arguments: {raw_args}, error: {e}")
                    tool_args = {}

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                tool_map = {t.name: t for t in tools}
                selected_tool = tool_map.get(tool_name)

                if selected_tool:
                    try:
                        # Call your tool
                        result = selected_tool.invoke(tool_args, user_id=user_id)
                        print(result)
                        return f"Tool `{tool_name}` executed successfully for  {tool_args.get('agent_name')}."

                    except Exception as e:
                        logger.error(f"Tool execution failed: {str(e)}")
                        return f"Tool `{tool_name}` execution failed: {str(e)}"
                        # internal_messages.append(
                        #     {"role": "tool_calling", "content": f" Tool `{tool_name}` execution failed: {str(e)}","ignore": True}
                        # )
        
        #     return Command(
        #         goto = "report_generation",
        #         update = {"messages": messages, "internal_messages": internal_messages}
        #     )

        # else:
        #     return state  # No tool calls, return state unchanged

    def tool_condition(self, state: state) -> bool:
        last_msg = state.messages[-1] if state.messages else None
        # Check if the state has any tool calls

        if "tool_calls" in last_msg.get("additional_kwargs", {}):
            print("executing tool")
            return "Tool_execution"
        else:
            return "No_tool_execution"

    def kpi_route_check(self, state: state):
        print("IN THE KPI ROUTE CHECK CONDITIONAL EDGE \n")
        agent_names = state.agent_names

        for agent in agent_names:
            kpi = get_agent_kpi(agent, user_id=state.user_id)
            if not kpi:  # If even one agent has no KPI
                print(f" {agent} does not have KPIs.")
                return "get_kpis"

        return "intent_classifier"

    def check_if_call(self,state: state):
        if state.call:
            return "report_generation"
        return END

    def check_pipeline(self,state):
        if state.has_run_pipelines:
            return "report_generation"
        return END

    def check_intent(self, state):
        intents = state.user_intents
        
        
        # If we have multiple parallel intents, route to multi-processor
        if state.parallel_intents:
            return "Multi_intent_processor"
        
        # Single intent routing (existing logic)
        if intents.team_information:
            return "get_kpis"
        
        elif intents.call_on_demand:
            return "call_on_demand"
        
        elif intents.call_managers:
            return "call_managers"
        
        elif intents.simple_response.simple_response or intents.simple_response.follow_up_question:
            return "simple_response"
        
        elif intents.business_rag_response:
            return "simple_response"  # Single RAG intent
        
        elif intents.agent_analysis:
            return "agent_response_analysis"  # Single analysis intent
        
        else:
            return "END"

    def build_graph(self) -> StateGraph:

        builder = StateGraph(state)
        builder.add_node("Tool_execution", self.tool_execution_node)
        builder.add_node("get_kpis", self.get_kpis)
        builder.add_node("update_messages", self.update_messages)
        builder.add_node("CAIO", self.CAIO)
        builder.add_node("report_generation", self.report_generation)
        builder.add_node("Intelligent_update", self.business_update)
        builder.add_node("intent_classifier", self.intent_classifier)
        builder.add_node("agent_response_analysis", self.agent_response_analysis)
        builder.add_node("handle_simple_response", self.handle_simple_response)
        builder.add_node("Multi_intent_processor", self.process_multiple_intents)

        builder.add_edge(START, "update_messages")
        builder.add_edge("update_messages", "Intelligent_update")
        builder.add_edge("Intelligent_update", "intent_classifier")
        # builder.add_conditional_edges("update_messages", self.kpi_route_check, {"get_kpis": "get_kpis", "intent_classifier": "intent_classifier"})
        builder.add_conditional_edges("intent_classifier", self.check_intent, {
            "Multi_intent_processor": "Multi_intent_processor",
            "get_kpis": "get_kpis",
            "call_on_demand": "CAIO",
            "call_managers": "CAIO",
            "set_or_update_scheduled_time": "CAIO",
            "simple_response": "handle_simple_response",
            "agent_response_analysis": "agent_response_analysis",
            "END": END
        })

        builder.add_edge("Multi_intent_processor", "report_generation")
        builder.add_edge("CAIO", "report_generation")
        builder.add_edge("agent_response_analysis", "report_generation")
        builder.add_edge("handle_simple_response", END)

        graph = builder.compile()
        return graph

if __name__ == "__main__":
    import asyncio
    async def main():
        caio = CAIO_Agent()
        graph = caio.graph
        current_state = {"query": "", "agent_names": [], "manager_names": [], "messages": [], "has_set_kpis": True, "Q_A": True}
        while True:
            # Get user input
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye!")
                break

            # Update the query in state
            current_state["query"] = user_input

            # Run graph with the updated state
            current_state = await graph.ainvoke(current_state)

            # Get assistant response
            if current_state["messages"]:
                assistant_msg = current_state["messages"][-1]["content"]
                print(f"Assistant: {assistant_msg}")
            else:
                print("Assistant: (no response)")
    asyncio.run(main())
