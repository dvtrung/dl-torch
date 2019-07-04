const ssh2 = require('ssh2');
const path = require('path');
const { Client } = ssh2;

export function sshConnect(machine, successCallback, errorCallback) {
  if ('proxy' in machine) {
    const client = new Client();
    const client2 = new Client();
    client.on('ready', () => {
      client.exec(`nc ${machine.host} 22`, (err, stream) => {
        if (err) {
          errorCallback(err);
          client.end();
        }
        client2.on('ready', () => {
          if (successCallback) successCallback(client2);
        }).on('error', (err) => {
          errorCallback(err);
          client.end()
        }).connect({
          sock: stream,
          username: machine.username,
          privateKey: machine.privateKey
        })
      })
    }).on('error', (err) => {
      errorCallback(err);
    }).connect({
      host: machine.proxy.host,
      port: machine.proxy.port || 22,
      username: machine.proxy.username,
      privateKey: machine.proxy.privateKey
    });
  } else {
    const client = new Client();
    client.on('ready', () => {
      if (successCallback) successCallback(client);
    }).on('error', (err) => {
      errorCallback(err)
    }).connect({
      host: machine.host,
      port: machine.port || 22,
      username: machine.username,
      privateKey: machine.privateKey
    })
  }
}

export function getLogPath(model) {
  const configName = path.basename(model.path).slice(0, -4);
  const logPath = `${model.remoteDir}/logs/${configName}`
  return `${logPath}/\$(ls ${logPath} -S | tail -n 1)`
}
