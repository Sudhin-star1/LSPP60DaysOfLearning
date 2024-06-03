const { kafka } = require("./client");

async function init() {
  const admin = kafka.admin();
  console.log("Admin Connecting...");
  admin.connect();
  console.log("Admin Connecting Success...");
  console.log("Creating Topics rider-updated...");
  admin.createTopics({
    topics: [
      {
        topic: "rider-updated",
        numPartitions: 2,
      },
    ],
  });
  console.log("Topics Created rider-updated...");

  console.log("Disconnecting Admin!!!");
  await admin.disconnect();
}

init();



