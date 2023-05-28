import React from 'react'
import styles from './Instruction.module.scss';
import instruction from '../../../assets/instruction.jpeg';
import CustomButton from '../../../components/CustomButton/CustomButton';

const Instruction = ({closeModal}) => {

  const handleStart = () => {
      closeModal();
  };
  return (
    <div style={{display: 'flex', flexDirection: 'column'}}>
        <img className={styles.visible} src={instruction} alt=""/>

        <CustomButton name='Начать обход квартиры' handleClick={handleStart}/>
    </div>
  )
}

export default Instruction;