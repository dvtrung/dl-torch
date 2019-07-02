// @flow
import type { GetState, Dispatch } from '../reducers/types';

const path = require('path');
const fs = require('fs');
const ssh2 = require('ssh2');
const { Client } = ssh2

export const LOAD_DIRECTORY = 'LOAD_DIRECTORY';
export const SELECT_MODEL = 'SELECT_MODEL';

export function loadModelsAsync(path) {
  return (dispatch: Dispatch) => {
    fs.readdir(`${path}\\model_configs`, function (err, filenames) {
      if (err) {
          return console.log('Unable to scan directory: ' + err);
      }
      
      const models = {};
      filenames.forEach(filename => {
        const name = filename.substr(0, filename.length - 4)
        models[name] = {
          path: filename
        };
      });

      dispatch({
        type: LOAD_DIRECTORY,
        name: path,
        path,
        models
      });
    });
  }
}

export function onModelSelected(root, modelName) {
  return (dispatch: Dispatch) => {
    const content = fs.readFileSync(
      path.join(root, 'model_configs', `${modelName}.yml`), 
      { encoding: 'utf-8' });
    dispatch({
      type: SELECT_MODEL,
      model: {
        modelName, root,
        config: content
      }
    });
  }
}

export function sshConnect(machine) {
  const conn = new Client();
  conn.on('ready', () => {
    console.log('Client :: ready');
    conn.exec('uptime', function(err, stream) {
      if (err) throw err;
      stream.on('close', function(code, signal) {
        console.log('Stream :: close :: code: ' + code + ', signal: ' + signal);
        conn.end();
      }).on('data', function(data) {
        console.log('STDOUT: ' + data);
      }).stderr.on('data', function(data) {
        console.log('STDERR: ' + data);
      });
    });
  }).connect({
    host: machine.host,
    port: machine.port || 22,
    username: machine.username,
    privateKey: machine.privateKey
  })
}
