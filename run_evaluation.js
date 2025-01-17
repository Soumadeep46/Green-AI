const { exec } = require('child_process');

console.log("Evaluating model and generating accuracy graph...");

exec('eval.py', (error, stdout, stderr) => {
  if (error) {
    console.error(`Error: ${error.message}`);
    return;
  }
  if (stderr) {
    console.error(`stderr: ${stderr}`);
    return;
  }
  console.log(`Evaluation output:\n${stdout}`);
  console.log("Evaluation completed. Check 'accuracy_graph.png' for the accuracy graph.");
});
