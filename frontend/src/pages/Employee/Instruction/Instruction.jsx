import React from 'react'
import styles from './Instruction.module.scss';
import instruction from '../../../assets/instruction.png';
import CustomButton from '../../../components/CustomButton/CustomButton';

const Instruction = ({closeModal}) => {

  const handleStart = () => {
      closeModal();
  };

  return (
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <img className={styles.visible} src={instruction} alt=""/>

        <CustomButton name='Начать обход квартиры' width={'185px'} handleClick={handleStart}/>
    </div>
  )
}

export default Instruction;