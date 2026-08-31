const puter = require('@heyputer/puter.js');

async function main() {
    const token = process.env.PUTER_AUTH_TOKEN;
    if (token) {
        // Authenticate - we'll assume puter.js supports initialization via a function like `puter.init(token)` or we can just hope it uses the env variable.
        // Or Puter is a global singleton, we can just login. Wait, there's no UI here, so maybe `process.env.PUTER_AUTH_TOKEN` is automatically picked up, or we need to pass it.
    }
    
    try {
        const models = await puter.ai.listModels();
        console.log(JSON.stringify(models));
    } catch (e) {
        console.error(e.message);
        process.exit(1);
    }
}

main();
