# Project Reflection

## Design Decision

A key design decision was to use a tool-based architecture where the AI agent delegates specialized tasks to dedicated tools instead of handling every request directly. The agent uses MCP Gateway tools for customer and order operations, a Knowledge Base tool for retrieving product and loyalty information, long-term memory for maintaining customer context, Code Interpreter for loyalty discount calculations, and the Browser tool for accessing live web information. This separation makes the system easier to extend and allows each capability to be tested independently.

## Challenge and Resolution

One of the main challenges was integrating the different AgentCore services while ensuring that the deployed runtime had the required permissions. The Knowledge Base initially failed because the runtime role did not have permission to call the Bedrock Retrieve API. I resolved this by adding an IAM policy allowing `bedrock:Retrieve` on the project's Knowledge Base. Another challenge occurred with the Browser tool, where the initial runtime configuration did not successfully authorize browser operations. I resolved this by configuring the AgentCore Browser correctly and redeploying the runtime. After the changes, the browser successfully accessed a live Udacity webpage and returned its title.

## Production Consideration

For production, security and observability would be major considerations. IAM permissions should follow least-privilege principles rather than granting broad access. Customer information stored in long-term memory should also be protected and governed with appropriate retention and access controls. In addition, CloudWatch logging, monitoring, error tracking, and usage metrics should be used to identify failures and control operational costs. The system should also include validation and rate limiting for external tools and APIs.
