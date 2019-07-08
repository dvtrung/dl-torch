// @flow
import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import routes from '../constants/routes';
import styles from './Home.css';
import {Statistic, Row, Col, Icon, Tooltip} from 'antd';
type Props = {
  stats: any,
  stepStats: any
};

export default class Stats extends Component<Props> {
  constructor() {
    super();
    this.state = {
      currentMetric: 0
    }
  }

  onSelect = (keys, event) => {
    console.log('Trigger Select', keys, event);
  };

  onExpand = () => {
    console.log('Trigger Expand');
  };

  render() {
    const { stats, stepStats } = this.props;
    const { currentMetric } = this.state;
    const metric = stats.metrics ? stats.metrics[currentMetric] : null;
    return [
      <Row gutter={16} key="epoch">
        <Col span={8}>
          <Statistic title="Epoch" value={stats.epoch || 0} suffix={"/ " + (stats.totalEpoch || 0)} />
        </Col>
        <Col span={8}>
          <Statistic
            title={`Train ${metric || ""}`}
            value={metric ? (stats.bestResult[this.state.currentMetric] * 100) : ""}
            precision={2} />
        </Col>
        <Col span={8}>
          <Tooltip title={metric ? `Epoch ${stats.bestResultEpoch[this.state.currentMetric]}`: ""} placement="bottom">
            <span>
            <Statistic
              title={`Test ${metric || ""}`}
              value={metric ? (stats.bestResult[this.state.currentMetric] * 100) : ""}
              precision={2} />
            </span>
          </Tooltip>
        </Col>
      </Row>,
      <Row gutter={16} key="epoch-step">
        <Col span={8}>
          <Statistic title="Epoch progress" value={stepStats.epoch * 100 || 0} suffix={"%"} precision={2} />
        </Col>
        <Col span={8}>
          <Statistic
            title={`Training Loss`}
            value={stepStats.overallLoss || 0}
            precision={2} />
        </Col>
        <Col span={8}>
          <Tooltip title={metric ? `Epoch ${stats.bestResultEpoch[this.state.currentMetric]}`: ""} placement="bottom">
            <span>
            <Statistic
              title={`Test ${metric || ""}`}
              value={metric ? (stats.bestResult[this.state.currentMetric] * 100) : ""}
              precision={2} />
            </span>
          </Tooltip>
        </Col>
      </Row>
    ];
  }
}

