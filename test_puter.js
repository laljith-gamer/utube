const puterRaw = require('@heyputer/puter.js');
const puter = puterRaw.puter || puterRaw.default || puterRaw;

async function main() {
    const token = process.env.PUTER_AUTH_TOKEN;
    if (token) {
        puter.authToken = token;
        console.log("authenticated: true");
    } else {
        console.log("authenticated: false");
    }
    
    try {
        const models = await puter.ai.listModels();
        console.log("Puter connected successfully. Models count: " + (models ? models.length : 0));
    } catch (e) {
        console.error("Puter connection failed: " + e.message);
        process.exit(1);
    }
}

main();
