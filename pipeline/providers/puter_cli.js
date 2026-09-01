const puterRaw = require('@heyputer/puter.js');
const puter = puterRaw.puter || puterRaw.default || puterRaw;

if (process.env.PUTER_AUTH_TOKEN) {
    puter.authToken = process.env.PUTER_AUTH_TOKEN;
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error("Usage: node puter_cli.js <command> [json_payload]");
        process.exit(1);
    }
    
    const command = args[0];
    const payloadStr = args[1] || '{}';
    let payload = {};
    try {
        payload = JSON.parse(payloadStr);
    } catch (e) {
        console.error("Invalid JSON payload");
        process.exit(1);
    }

    try {
        if (command === 'listModels') {
            const models = await puter.ai.listModels();
            console.log(JSON.stringify(models));
        } else if (command === 'chat') {
            const { model, messages, stream } = payload;
            
            // Reformat messages to what puter expects (if it differs from standard)
            // Note: puter.ai.chat supports standard openai-like params.
            const response = await puter.ai.chat(messages, {
                model: model,
                stream: !!stream
            });

            let text = "";
            let finishReason = "unknown";
            let usage = {};

            if (response.message && response.message.content) {
                let content = response.message.content;
                if (Array.isArray(content)) {
                    text = content.filter(b => b.type === 'text').map(b => b.text).join('');
                } else {
                    text = content;
                }
                finishReason = response.finish_reason || "stop";
            } else if (response.choices && response.choices.length > 0) {
                let content = response.choices[0].message.content;
                if (Array.isArray(content)) {
                    text = content.filter(b => b.type === 'text').map(b => b.text).join('');
                } else {
                    text = content;
                }
                finishReason = response.choices[0].finish_reason || "stop";
            } else {
                text = JSON.stringify(response);
            }

            if (response.usage) {
                usage = response.usage;
            }

            console.log(JSON.stringify({
                text: text,
                finishReason: finishReason,
                usage: usage,
                model: model,
                provider: "puter",
                raw: response
            }));
        } else {
            console.error(JSON.stringify({ error: "Unknown command: " + command }));
            process.exit(1);
        }
    } catch (e) {
        // Output error as JSON so python can parse it
        const errorMsg = e.message || String(e);
        const isRateLimit = errorMsg.includes("429") || 
                            errorMsg.toLowerCase().includes("rate limit") || 
                            e.status === 429 || 
                            (e.response && e.response.status === 429);
        console.error(JSON.stringify({ error: errorMsg, is_rate_limit: isRateLimit, raw_error: e }));
        process.exit(1);
    }
}

main();
