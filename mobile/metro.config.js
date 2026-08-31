const path = require("path");
const { getDefaultConfig } = require("expo/metro-config");

const projectRoot = __dirname;
const config = getDefaultConfig(projectRoot);

config.projectRoot = projectRoot;
config.watchFolders = [projectRoot];
config.resolver.blockList = [
  /.*\/data\/landmarks\/.*/,
  /.*\/data\/splits\/.*/,
  /.*\/checkpoints\/.*/,
  /.*\/notebooks\/.*/,
  /.*\/reports\/.*/
];

module.exports = config;
