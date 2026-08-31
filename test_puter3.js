const puter = require('@heyputer/puter.js').puter || require('@heyputer/puter.js').default;
console.log(Object.keys(puter));
console.log(puter.ai ? Object.keys(puter.ai) : "No puter.ai");
