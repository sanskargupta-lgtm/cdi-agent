"""
Client for calling Databricks Agent Endpoint via MLflow Deployments
Uses MLflow client which handles auth automatically via WorkspaceClient
"""
import json
import os
import time
from typing import Dict, Any
import requests
from mlflow.deployments import get_deploy_client



class AgentEndpointClient:
    """Client for calling Databricks Agent Endpoint via MLflow Deployments"""

    def __init__(self, agent_endpoint_name: str = None, workspace_client=None, endpoint_url: str = None, access_token: str = None):
        """
        Initialize agent endpoint client

        Args:
            agent_endpoint_name: Name of the deployed agent endpoint (optional if endpoint_url provided)
            workspace_client: Databricks WorkspaceClient instance (optional if endpoint_url provided)
            endpoint_url: Direct HTTP URL to the agent endpoint (optional)
            access_token: Databricks access token for authentication (optional)
        """
        # Support both MLflow Deployments and direct HTTP access
        self.use_direct_http = bool(endpoint_url)
        
        if self.use_direct_http:
            self.endpoint_url = endpoint_url
            self.access_token = access_token
            print(f"[AgentClient] Initialized with direct HTTP endpoint: {endpoint_url}")
        else:
            self.agent_endpoint_name = agent_endpoint_name
            self.deploy_client = get_deploy_client("databricks")
            print(f"[AgentClient] Initialized with MLflow endpoint: {agent_endpoint_name}")
    
    def query_stream(self, user_message: str, user_email: str = None, session_id: str = None):
        """
        Stream response from agent endpoint with trace events support.
        
        Args:
            user_message: User's question
            user_email: User's email for session tracking
            session_id: App-level session ID for conversation grouping
            
        Yields:
            JSON strings containing either trace events or final response
        """
        print(f"[AgentClient] Streaming message to agent endpoint...")
        print(f"[AgentClient] User: {user_email}")
        print(f"[AgentClient] Session ID: {session_id}")
        
        try:
            if not self.use_direct_http:
                print(f"[AgentClient] ERROR: Streaming only supported with direct HTTP access")
                error_response = {
                    "response": "Streaming not supported with MLflow Deployments client",
                    "charts": [],
                    "table_data": None,
                    "error": "Use direct HTTP endpoint URL for streaming"
                }
                yield json.dumps(error_response)
                return
            
            # Generate conversation ID
            import uuid
            conversation_id = str(uuid.uuid4())
            
            # Direct HTTP call to agent endpoint with streaming
           
            # decoded_token = jwt.decode(self.access_token, options={"verify_signature": False})
            # print(f"token used for calling endpooint {decoded_token}")
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            
            # Databricks playground format
            payload = {
                "input": [{"role": "user", "content": user_message}],
                "custom_inputs": {
                    "conversation_id": conversation_id,
                    "user_id": user_email or "anonymous@crunchyroll.com",
                    "session_id": session_id,  # Pass app session ID to agent
                    "user_token": self.access_token  # Pass user token for Genie API auth
                },
                "databricks_options": {
                    "return_trace": True
                },
                "stream": True
            }
            
            print(f"[AgentClient] Making streaming HTTP POST to {self.endpoint_url}")
            print(f"[AgentClient] Payload: {json.dumps(payload)}")
            
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                stream=True,  # Enable streaming
                timeout=300
            )
            
            print(f"[AgentClient] Response status code: {response.status_code}")
            response.raise_for_status()
            
            # Process Server-Sent Events (SSE) stream
            print(f"[AgentClient] Processing SSE stream")
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8').strip()
                        
                        # Skip empty lines
                        if not decoded_line:
                            continue
                        
                        # Handle SSE format: "data: {json}"
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[6:]  # Remove "data: " prefix
                            
                            # Check for [DONE] marker
                            if json_str == '[DONE]':
                                print(f"[AgentClient] Stream completed")
                                break
                            
                            try:
                                chunk_data = json.loads(json_str)
                                
                                # Extract delta from response.output_text.delta events
                                if chunk_data.get("type") == "response.output_text.delta":
                                    delta = chunk_data.get("delta", "")
                                    if delta:
                                        try:
                                            # Parse delta as JSON
                                            obj = json.loads(delta)
                                            
                                            # Yield traces
                                            if obj.get("type") == "trace":
                                                print(f"[AgentClient] Trace: {obj.get('step')} - {obj.get('status')}")
                                                yield json.dumps(obj)
                                                time.sleep(0.1)  # Small delay for visual effect
                                            
                                            # Yield final response
                                            elif "response" in obj:
                                                print(f"[AgentClient] Final response received")
                                                yield json.dumps(obj)
                                        
                                        except json.JSONDecodeError:
                                            # Delta is not JSON, skip
                                            pass
                                
                            except json.JSONDecodeError as e:
                                print(f"[AgentClient] JSON decode error: {e}")
                                continue
                        
                    except Exception as e:
                        print(f"[AgentClient] Error processing line: {e}")
                        continue
        
        except requests.exceptions.RequestException as e:
            error_response = {
                "response": f"Connection error: {str(e)}",
                "charts": [],
                "table_data": None,
                "error": str(e)
            }
            yield json.dumps(error_response)

    def chat(self, user_message: str, user_email: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Send a message to the agent endpoint

        Args:
            user_message: User's question
            user_email: User's email for session tracking and logging
            session_id: App-level session ID for conversation grouping

        Returns:
            Dict with 'response', 'messages', 'charts', 'table_data', and 'error' keys
        """
        print(f"[AgentClient] Sending message to agent endpoint...")
        print(f"[AgentClient] User: {user_email}")
        print(f"[AgentClient] Session ID: {session_id}")

        try:
            # Prepare custom inputs with user information
            custom_inputs = {}
            if user_email:
                custom_inputs["user_id"] = user_email
            
            # Use provided session_id from app (created by AppSessionManager)
            if session_id:
                custom_inputs["session_id"] = session_id
            elif user_email:
                # Fallback only if no session_id provided
                custom_inputs["session_id"] = f"session_{user_email}"
            
            if self.use_direct_http:
                # Direct HTTP call to agent endpoint
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "input": [{"role": "user", "content": user_message}]
                }
                
                # Add custom_inputs if present
                if custom_inputs:
                    payload["custom_inputs"] = custom_inputs
                
                print(f"[AgentClient] Making HTTP POST to {self.endpoint_url}")
                print(f"[AgentClient] Request payload: {json.dumps(payload, indent=2)}")
                
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                
                print(f"[AgentClient] Response status code: {response.status_code}")
                
                response.raise_for_status()
                response_data = response.json()
                
            else:
                # Call agent endpoint with user message using MLflow Deployments Client
                # Agent endpoint expects "input" (array of messages), not "messages"
                inputs = {"input": [{"role": "user", "content": user_message}]}
                
                # Add custom_inputs if present
                if custom_inputs:
                    inputs["custom_inputs"] = custom_inputs
                
                response_data = self.deploy_client.predict(
                    endpoint=self.agent_endpoint_name,
                    inputs=inputs,
                )

            print(f"[AgentClient] Received response from agent endpoint")
            print(f"[AgentClient] Raw response type: {type(response_data)}")
            print(
                f"[AgentClient] Raw response keys: {response_data.keys() if isinstance(response_data, dict) else 'Not a dict'}"
            )
            if isinstance(response_data, dict):
                print(f"[AgentClient] Response sample: {str(response_data)[:500]}")

            # Parse agent response
            result = self._parse_agent_response(response_data)
            print(
                f"[AgentClient] Parsed response: {len(result.get('charts', []))} charts, table_data: {bool(result.get('table_data'))}"
            )

            return result

        except Exception as e:
            error_msg = f"Error calling agent endpoint: {str(e)}"
            print(f"[AgentClient] ERROR: {error_msg}")
            import traceback

            traceback.print_exc()
            return {
                "response": None,
                "messages": [],
                "charts": [],
                "table_data": None,
                "error": error_msg,
            }

    def _parse_agent_response(self, raw_response: dict) -> Dict[str, Any]:
        """
        Parse agent endpoint response to extract summary, charts, and table data

        Agent response format: {"object": "response", "output": [...], "id": "..."}
        """
        output_array = raw_response.get("output", [])
        print(f"[AgentClient] Parsing {len(output_array)} output items...")

        # First, try to detect if the agent returned a JSON-encoded response
        # This happens when the agent streams: {"content": "{\"response\": ..., \"charts\": [...], ...}"}
        for item in output_array:
            if item.get("type") == "text" or item.get("type") == "message":
                text_content = self._get_text_from_item(item)
                if text_content and text_content.strip().startswith("{"):
                    try:
                        # Try to parse as JSON
                        parsed_data = json.loads(text_content)
                        if isinstance(parsed_data, dict) and "response" in parsed_data:
                            print(f"[AgentClient] Detected JSON-encoded response format")
                            return {
                                "response": parsed_data.get("response", ""),
                                "messages": output_array,
                                "charts": parsed_data.get("charts", []),
                                "table_data": parsed_data.get("table_data"),
                                "error": parsed_data.get("error"),
                            }
                    except json.JSONDecodeError:
                        pass  # Not JSON, continue with standard parsing

        # Standard parsing for structured output
        return {
            "response": self._extract_summary(output_array),
            "messages": output_array,
            "charts": self._extract_charts(output_array),
            "table_data": self._extract_genie_table(output_array),
            "error": None,
        }

    def _extract_summary(self, output_array: list) -> str:
        """Extract the final text summary from agent output"""
        for item in reversed(output_array):  # Start from end - summary is usually last
            if text := self._get_text_from_item(item):
                print(f"[AgentClient] Found summary: {text[:100]}...")
                return text

        return "Agent processed your request"

    def _get_text_from_item(self, item: dict) -> str:
        """Extract text from various item formats"""
        item_type = item.get("type")

        # Simple text type
        if item_type == "text":
            return item.get("text", "")

        # Message type with nested content
        if item_type == "message":
            content = item.get("content", "")

            # Content as string
            if isinstance(content, str):
                return content

            # Content as list of objects with 'text' field
            if isinstance(content, list) and content:
                first_item = content[0]
                if isinstance(first_item, dict):
                    return first_item.get("text", "")

        return ""

    def _extract_charts(self, output_array: list) -> list:
        """Extract all Plotly charts from function call outputs"""
        charts = []

        for item in output_array:
            if item.get("type") != "function_call_output":
                continue

            if chart := self._parse_chart_output(item.get("output", "")):
                charts.append(chart)
                print(
                    f"[AgentClient] Found chart: {chart.get('chart_type', 'unknown')}"
                )

        return charts

    def _parse_chart_output(self, output_str: str) -> dict:
        """Parse a single function output for chart data"""
        try:
            data = json.loads(output_str) if isinstance(output_str, str) else output_str

            if not isinstance(data, dict):
                return None

            # Direct chart format: {"plotly_json": {...}, "chart_type": "..."}
            if "plotly_json" in data:
                return data

            # UC function wrapper format: {"rows": [["{...}"]], "columns": [...]}
            if "rows" in data and "columns" in data:
                if data["rows"] and data["rows"][0]:
                    chart_json = json.loads(data["rows"][0][0])
                    if "plotly_json" in chart_json:
                        return chart_json

            return None

        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
            print(f"[AgentClient] Error parsing chart: {e}")
            return None

    def _extract_genie_table(self, output_array: list) -> dict:
        """Extract table data from Genie function call outputs"""
        for item in output_array:
            if item.get("type") != "function_call_output":
                continue

            if table := self._parse_genie_output(item.get("output", "")):
                print(f"[AgentClient] Found Genie table: {len(table['data'])} rows")
                return table

        return None

    def _parse_genie_output(self, output_str: str) -> dict:
        """Parse Genie's complex response into simple table format"""
        try:
            # Parse outer JSON
            data = json.loads(output_str) if isinstance(output_str, str) else output_str

            if not isinstance(data, dict) or "content" not in data:
                return None

            # Parse nested content
            content = data["content"]
            content_data = json.loads(content) if isinstance(content, str) else content

            # Check for Genie statement_response
            if "statement_response" not in content_data:
                return None

            # Transform complex format to simple format
            return self._transform_genie_response(content_data)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[AgentClient] Error parsing Genie output: {e}")
            return None

    def _transform_genie_response(self, genie_response: dict) -> dict:
        """
        Transform Genie's nested response to simple table format

        FROM: {"statement_response": {"result": {"data_array": [...]}, "manifest": {...}}}
        TO:   {"data": [[...]], "columns": [...]}
        """
        try:
            stmt_resp = genie_response["statement_response"]
            result = stmt_resp["result"]
            schema = stmt_resp["manifest"]["schema"]

            # Extract columns
            columns = [col["name"] for col in schema["columns"]]

            # Extract data rows
            data = [
                [val["string_value"] for val in row["values"]]
                for row in result["data_array"]
            ]

            return {"data": data, "columns": columns}

        except (KeyError, TypeError) as e:
            print(f"[AgentClient] Error transforming Genie response: {e}")
            return None
