const {ocr} = require("llama-ocr")

async function runOCR(filePath, apiKey){
    try {
        const markdown = await ocr({ filePath, apiKey});
        console.log(markdown);

    } catch (error) {
        console.error("Error:", error);
    }
}

runOCR(process.argv[2], process.argv[3]);