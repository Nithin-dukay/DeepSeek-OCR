const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.send('<h1>Hello World!</h1><p>Welcome to your simple Hello World app.</p>');
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
  console.log('Press Ctrl+C to stop the server');
});
