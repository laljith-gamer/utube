const puterRaw = require('@heyputer/puter.js');
const puter = puterRaw.puter || puterRaw.default || puterRaw;

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
            console.log(JSON.stringify(response));
        } else {
            console.error("Unknown command: " + command);
            process.exit(1);
        }
    } catch (e) {
        // Output error as JSON so python can parse it
        console.error(JSON.stringify({ error: e.message || String(e) }));
        process.exit(1);
    }
}

main();
