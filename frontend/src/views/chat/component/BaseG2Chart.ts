import { BaseChart } from '@/views/chat/component/BaseChart.ts'
import { Chart } from '@antv/g2'
import mitt from 'mitt'

// 创建全局事件发射器用于图表点击事件
export const chartEventEmitter = mitt()

// 图表点击事件数据类型
export interface ChartClickEventData {
  chartType: string
  chartId: string
  rawData: any
  x: {
    field: string
    name: string
    value: any
  } | null
  y: {
    field: string
    name: string
    value: any
  } | null
  series: {
    field: string
    name: string
    value: any
  } | null
}

export abstract class BaseG2Chart extends BaseChart {
  chart: Chart

  constructor(id: string, name: string) {
    super(id, name)
    this.chart = new Chart({
      container: id,
      autoFit: true,
      padding: 'auto',
    })

    this.chart.theme({
      view: {
        viewFill: '#FFFFFF',
      },
    })
  }

  render() {
    this.chart?.render()
    // 添加点击事件监听
    this.bindClickEvent()
  }

  /**
   * 绑定图表点击事件，获取点击区域的 x/y 轴数据
   */
  private bindClickEvent() {
    // 监听图表元素点击事件
    this.chart.on('element:click', (event: any) => {
      const { data } = event
      if (data) {
        // 获取 x 轴和 y 轴对应的字段
        const xAxisField = this.axis.find((item) => item.type === 'x')
        const yAxisField = this.axis.find((item) => item.type === 'y')
        const seriesField = this.axis.find((item) => item.type === 'series')

        const clickedData: ChartClickEventData = {
          chartType: this._name,
          chartId: this.id,
          // 原始数据
          rawData: data.data || data,
          // x 轴数据
          x: xAxisField
            ? {
                field: xAxisField.value,
                name: xAxisField.name,
                value: (data.data || data)[xAxisField.value],
              }
            : null,
          // y 轴数据
          y: yAxisField
            ? {
                field: yAxisField.value,
                name: yAxisField.name,
                value: (data.data || data)[yAxisField.value],
              }
            : null,
          // 系列数据（如果有）
          series: seriesField
            ? {
                field: seriesField.value,
                name: seriesField.name,
                value: (data.data || data)[seriesField.value],
              }
            : null,
        }

        console.log('=== 图表点击事件 ===')
        console.log('图表类型:', this._name)
        console.log('点击数据:', clickedData)
        console.log('X轴:', clickedData.x?.name, '=', clickedData.x?.value)
        console.log('Y轴:', clickedData.y?.name, '=', clickedData.y?.value)
        if (clickedData.series) {
          console.log('系列:', clickedData.series?.name, '=', clickedData.series?.value)
        }
        console.log('原始数据:', clickedData.rawData)
        console.log('==================')

        // 发射图表点击事件，用于触发文章详情弹窗
        chartEventEmitter.emit('chart-click', clickedData)
      }
    })
  }

  destroy() {
    // 移除事件监听
    this.chart?.off('element:click')
    this.chart?.destroy()
  }
}
